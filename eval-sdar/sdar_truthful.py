"""
SDAR TruthfulQA MC2 Evaluation (Masked Prediction Version).

Logic:
1. Input: [Prompt] + [MASK] * len(Choice)
2. Target: [Choice]
3. Attention: SDAR Block Attention (allows MASKs to see Prompt and each other within block)
4. Score: CrossEntropy(Mask_Logits, Choice_Labels)
"""

import os
import sys
import json
import shutil
# [Fix] Complete missing imports
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F
from datasets import load_dataset
from termcolor import cprint
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from accelerate import Accelerator

# =========================================================================
# [SETUP] Import Models
# =========================================================================
PROJECT_ROOT = "your_path/project_root"
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from models import SDARForCausalLM
except ImportError:
    cprint("Warning: Could not import SDARForCausalLM, using AutoModel.", "red")
    from transformers import AutoModelForCausalLM as SDARForCausalLM

# -------------------------------------------------------------------------
# SDAR Attention Utilities
# -------------------------------------------------------------------------
def make_eval_block_attention(N: int, start_pos: int, block_size: int) -> torch.Tensor:
    """
    Constructs SDAR Block Attention Mask.
    Allows MASK tokens to attend to Prompt (0..L0) and previous/current blocks.
    """
    B = 1
    L0 = start_pos
    L1 = N - start_pos
    
    bias = torch.full((B, 1, N, N), 0)
    
    # Prompt part (Causal)
    tril = torch.tril(torch.ones(L0, L0))
    bias[:, :, :L0, :L0] = tril.unsqueeze(0).unsqueeze(0)
    
    # Response (Masked) part
    rows = torch.arange(L0, N)
    num_blocks = (L1 + block_size - 1) // block_size
    
    for bi in range(num_blocks):
        local_start = bi * block_size
        local_end = min((bi + 1) * block_size, L1)
        col_start = L0 + local_start
        col_end = L0 + local_end
        
        block_rows = torch.arange(col_start, col_end)
        
        # 1. See Prompt (0 ~ L0)
        bias[:, :, block_rows.unsqueeze(-1), 0:L0] = 1
        
        # 2. See previous Response Blocks
        if col_start > L0:
            bias[:, :, block_rows.unsqueeze(-1), L0:col_start] = 1
            
        # 3. See current Block internally (SDAR feature: Block parallel)
        bias[:, :, block_rows.unsqueeze(-1), col_start:col_end] = 1

    return bias

def process_pad(attn: torch.Tensor, input_ids: torch.Tensor, start_pos: int, pad_id: int) -> torch.Tensor:
    key_mask = (input_ids == pad_id).unsqueeze(1).unsqueeze(1)
    attn.masked_fill_(key_mask, 0)
    return attn

# -------------------------
# Script Arguments
# -------------------------
@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to SDAR model."})
    output_dir: str = field(default="results_truthful_mc2", metadata={"help": "Where to write results."})
    max_examples: Optional[int] = field(default=None, metadata={"help": "Cap for quick debug."})
    block_size: int = field(default=4, metadata={"help": "SDAR block size."})
    dtype: str = field(default="bf16", metadata={"help": "bf16 or fp16"})

def make_prompt_zeroshot(question: str, tokenizer) -> str:

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        messages = [{"role": "user", "content": question}]
        # Only generate Prompt part, don't add Generation Prompt
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except:
            return f"User: {question}\nAssistant:"
    
    # Fallback default
    return f"Q: {question}\nA:"

@torch.no_grad()
def sdar_choice_score(
    model,
    tokenizer,
    prompt: str,
    choice: str,
    block_size: int,
    device: torch.device,
) -> float:
    # 1. Tokenize
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    choice_ids = tokenizer(" " + choice, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    
    L0 = prompt_ids.size(1)
    L1 = choice_ids.size(1)
    if L1 == 0: return float("-inf")
    
    # 2. Construct [Prompt, MASK_CHOICE]
    # Must mask Choice, otherwise SDAR Attention will see the answer (Data Leakage)
    mask_token_id = tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
    masked_choice = torch.full_like(choice_ids, mask_token_id)
    
    # Input: [Prompt, MASK, MASK...]
    input_ids = torch.cat([prompt_ids, masked_choice], dim=1)
    L = input_ids.size(1)
    
    # 3. Attention Mask
    attn = make_eval_block_attention(L, L0, block_size).to(device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    attn = process_pad(attn, input_ids, L0, pad_id)
    
    # 4. Position IDs
    position_ids = torch.arange(L, device=device).unsqueeze(0)
    
    # 5. Forward
    outputs = model(
        input_ids=input_ids,
        attention_mask=attn,
        position_ids=position_ids,
        return_dict=True
    )
    
    logits = outputs.logits # [1, L, V]
    if isinstance(logits, tuple): logits = logits[0]
    
    # 6. Score: Predict Choice from MASK
    # Logits at [L0:] correspond to predictions for positions [L0:]
    choice_logits = logits[:, L0:, :] # [1, L1, V]
    
    # Targets are the REAL choice_ids
    choice_labels = choice_ids # [1, L1]
    
    log_probs = F.log_softmax(choice_logits, dim=-1)
    token_logp = log_probs.gather(dim=-1, index=choice_labels.unsqueeze(-1)).squeeze(-1)
    
    # 7. Length Norm
    sum_logp = token_logp.sum(dim=-1)
    avg_logp = sum_logp / max(L1, 1)
    
    return float(avg_logp.item())

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    accelerator = Accelerator()
    
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        cprint("Loading TruthfulQA...", "yellow")

    # Cache Fix
    local_cache_dir = os.path.join(args.output_dir, "hf_cache")
    try:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation", 
                          cache_dir=local_cache_dir, verification_mode="no_checks")
    except Exception:
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation", 
                          download_mode="force_redownload")

    if args.max_examples:
        ds = ds.select(range(min(args.max_examples, len(ds))))

    # Load Model
    if accelerator.is_main_process:
        cprint(f"Loading Model: {args.model_name_or_path}", "yellow")
        
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    model = SDARForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    model = accelerator.prepare(model)
    model.eval()

    out_file = os.path.join(args.output_dir, "truthful_mc2_results.jsonl")
    pm_true_list = []

    # Simple Main Process Loop (Efficiency isn't bottleneck here)
    if accelerator.is_main_process:
        with open(out_file, "w", encoding="utf-8") as f:
            for ex in tqdm(ds, desc="Eval"):
                q = ex["question"]
                choices = ex["mc2_targets"]["choices"]
                labels = ex["mc2_targets"]["labels"]
                
                # Use smarter prompt construction
                prompt = make_prompt_zeroshot(q, tokenizer)
                scores = []
                
                for ch in choices:
                    score = sdar_choice_score(
                        model=model, 
                        tokenizer=tokenizer, 
                        prompt=prompt, 
                        choice=ch, 
                        block_size=args.block_size,
                        device=accelerator.device
                    )
                    scores.append(score)
                
                scores_t = torch.tensor(scores)
                probs = torch.softmax(scores_t, dim=0) 
                labels_t = torch.tensor(labels, dtype=torch.bool)
                
                if labels_t.sum() > 0:
                    pm_true = probs[labels_t].sum().item()
                else:
                    pm_true = 0.0
                
                pm_true_list.append(pm_true)
                
                f.write(json.dumps({
                    "question": q, 
                    "pm_true": pm_true, 
                    "scores": scores
                }) + "\n")

        acc = sum(pm_true_list) / len(pm_true_list)
        cprint(f"✅ Final MC2 Accuracy: {acc:.4f}", "green")
        
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump({"acc": acc, "model": args.model_name_or_path}, f, indent=2)

if __name__ == "__main__":
    main()
