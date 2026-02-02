import os
import sys
import json
import torch
import torch.nn.functional as F
from tqdm import tqdm
from termcolor import cprint
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser
from accelerate import Accelerator
from datasets import load_dataset

# Assume dream package is in current path or python path
# Import DreamModel and DreamTokenizer based on provided code
try:
    from dream import DreamTokenizer
    from dream.modeling_dream import DreamModel
except ImportError:
    cprint("Error: Can't import dream. Make sure 'dream' folder is in PYTHONPATH.", "red")
    raise

# =========================================================================
# Configuration Parameters
# =========================================================================
@dataclass
class ScriptArguments:
    model_path: str = field(metadata={"help": "Path to the pretrained Dream model."})
    output_dir: str = field(default="results_dream_truthful", metadata={"help": "Output directory."})
    max_examples: Optional[int] = field(default=None, metadata={"help": "Debug limit."})
    batch_size: int = field(default=1, metadata={"help": "Eval batch size (keep 1 for safety in MC tasks)."})

# =========================================================================
# Dream Model Scoring Core Function
# =========================================================================
@torch.no_grad()
def dream_choice_score(
    model: DreamModel,
    tokenizer: DreamTokenizer,
    prompt: str,
    choice: str,
    device: torch.device
) -> float:
    """
    Calculate P(Choice | Prompt) score.
    For Masked models, the logic is:
    Input: [Prompt] [MASK] [MASK] ... (length equals Choice)
    Target: [Choice]
    Score: Average Log_prob of Choice tokens at the Mask positions.
    """
    
    # 1. Tokenize Prompt and Choice
    # DreamTokenizer usually doesn't need special chat template for base scoring,
    # but if it's an Instruct model, it's better to add Chat format. Here we reuse the template logic from your code.
    # For generality, we do simple concatenation here, or you can manually add ChatML markers.
    
    # Simple concatenation logic (if model is Base model):
    # text = prompt 
    # Or ChatML (if model is Instruct):
    full_prompt_str = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    # Encode
    prompt_ids = tokenizer(full_prompt_str, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    choice_ids = tokenizer(choice, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
    
    # 2. Prepare Masked Input
    # Get special token ID
    mask_token_id = model.config.mask_token_id
    
    L_prompt = prompt_ids.size(1)
    L_choice = choice_ids.size(1)
    
    if L_choice == 0:
        return -float("inf")

    # Construct Input: Prompt IDs + [MASK] * Choice Length
    masked_choice_part = torch.full_like(choice_ids, mask_token_id)
    input_ids = torch.cat([prompt_ids, masked_choice_part], dim=1) # [1, L_total]
    
    # 3. Construct additional inputs needed by Dream model (attention_mask, tok_idx)
    # Based on the _sample function logic you provided:
    # tok_idx = attention_mask.long().cumsum(-1) - 1
    
    attention_mask = torch.ones_like(input_ids) # Full attention
    tok_idx = torch.arange(input_ids.size(1), device=device).unsqueeze(0) # [0, 1, 2, ...]
    
    # 4. Forward Pass
    # DreamModel's forward signature is usually (input_ids, attention_mask, tok_idx, ...)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        tok_idx=tok_idx,
        use_cache=False
    )
    
    logits = outputs.logits # [1, L_total, Vocab]
    
    # 5. Calculate score
    # We only care about predictions for the Mask part (i.e., the part after Prompt)
    # Logits correspond to predictions for tokens at current positions
    choice_logits = logits[:, L_prompt:, :] # [1, L_choice, Vocab]
    choice_labels = choice_ids              # [1, L_choice]
    
    # Calculate Log Softmax
    log_probs = F.log_softmax(choice_logits, dim=-1)
    
    # Gather probabilities of actual Choice Tokens
    # gather dim=-1, index shape must match
    token_log_probs = log_probs.gather(dim=-1, index=choice_labels.unsqueeze(-1)).squeeze(-1)
    
    # 6. Normalized score (Length Normalization)
    # This is standard practice for TruthfulQA, preventing long choices from scoring too low due to probability multiplication
    score = token_log_probs.sum() / L_choice
    
    return score.item()

# =========================================================================
# Main Function
# =========================================================================
def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    accelerator = Accelerator()
    
    # 1. Load model
    if accelerator.is_main_process:
        cprint(f"Loading Dream Model from: {args.model_path}", "yellow")
    
    # Must use trust_remote_code=True because it's a custom model
    tokenizer = DreamTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = DreamModel.from_pretrained(
        args.model_path, 
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16
    )
    
    model = accelerator.prepare(model)
    model.eval()
    
    # 2. Load dataset
    if accelerator.is_main_process:
        cprint("Loading TruthfulQA (MC2)...", "yellow")
        
    try:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation", verification_mode="no_checks")
    except:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")

    if args.max_examples:
        ds = ds.select(range(args.max_examples))
    
    output_file = os.path.join(args.output_dir, "dream_mc2_results.jsonl")
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    
    pm_true_list = []
    
    # 3. Evaluation loop
    # For simplicity, use single-process loop here (MC2 evaluation is usually fast, doesn't need complex distributed dataloader)
    if accelerator.is_main_process:
        with open(output_file, "w", encoding="utf-8") as f:
            for ex in tqdm(ds, desc="Evaluating"):
                question = ex["question"]
                choices = ex["mc2_targets"]["choices"]
                labels = ex["mc2_targets"]["labels"] # 0 or 1
                
                scores = []
                
                # Score each choice
                for choice in choices:
                    score = dream_choice_score(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=question,
                        choice=choice,
                        device=accelerator.device
                    )
                    scores.append(score)
                
                # Calculate MC2 accuracy
                # TruthfulQA MC2 definition: Normalized probability assigned to the set of true answers
                # Actually, standard practice is: if total probability of true_choices > total probability of false_choices (one interpretation)
                # But HuggingFace Leaderboard implementation usually:
                # Convert scores -> probability distribution, sum probabilities of all correct choices (assuming normalized over all choices)
                
                scores_t = torch.tensor(scores)
                probs = torch.softmax(scores_t, dim=0)
                labels_t = torch.tensor(labels, dtype=torch.bool)
                
                if labels_t.sum() > 0:
                    pm_true = probs[labels_t].sum().item()
                else:
                    pm_true = 0.0
                
                pm_true_list.append(pm_true)
                
                # Write results
                res = {
                    "question": question,
                    "pm_true": pm_true,
                    "probs": probs.tolist(),
                    "labels": labels
                }
                f.write(json.dumps(res) + "\n")
                
        # 4. Final statistics
        acc = sum(pm_true_list) / len(pm_true_list)
        cprint(f"\n========================================", "green")
        cprint(f"✅ Final TruthfulQA MC2 Accuracy: {acc:.4f}", "green")
        cprint(f"========================================", "green")
        
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"acc": acc, "model": args.model_path}, f, indent=2)

if __name__ == "__main__":
    main()
