from __future__ import annotations
import math, json, os, time, re, random
import asyncio
import torch
import torch.nn.functional as F
import transformers
import multiprocessing as mp
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Any, Dict
from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm
from termcolor import cprint
from transformers import HfArgumentParser, AutoTokenizer
from transformers.utils import ModelOutput
from datasets import Dataset
import torch.distributions as dists
from jinja2 import Template
from openai import AsyncOpenAI, RateLimitError
from omegaconf import OmegaConf

# --- Dream Model Imports ---
try:
    from dream import DreamTokenizer
    from dream.modeling_dream import DreamModel
    from dream.generation_utils_block import DreamGenerationMixin, DreamGenerationConfig
except ImportError:
    print("Warning: Dream modules not found. Ensure the 'dream' package is in your path.")

# ==========================================
# Part 1: Dream Core Generation Logic (完整版)
# ==========================================

def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits

def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, tar=None):
    logits = logits.float()
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    dist = dists.Categorical(logits=logits)
    x0 = dist.sample()
    probs = dist.probs
    if temperature > 0:
        target = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    else:
        target, x0 = probs.max(dim=-1)
    if tar == "confidence":
        return target, x0
    return target, x0

@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None

@torch.no_grad()
def _sample(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.LongTensor],
    generation_config: DreamGenerationConfig,
    block_length: Optional[int] = 32,
    use_cache: bool = False,
    further_horizon: int = 128,
    mask_token_id: int = 151666,
    eos_token_id: int = 151645,
    pad_token_id: int = 151643,
    pad_target_penalty: float = 1.0,
    unmask_threshold: float = 0.9,
    verbose: bool = False
) -> Union[DreamModelOutput, torch.LongTensor]:
    
    output_history = generation_config.output_history
    return_dict_in_generate = generation_config.return_dict_in_generate
    max_length = generation_config.max_length
    steps = generation_config.steps
    temperature = generation_config.temperature
    top_p = generation_config.top_p
    top_k = generation_config.top_k
    tar = generation_config.tar
    alg_temp = generation_config.alg_temp
    cgws = further_horizon
    
    histories = [] if (return_dict_in_generate and output_history) else None
    
    # Init sequences with mask tokens
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    gen_length = max_length - input_ids.shape[1]
    
    if block_length is None:
        block_length = gen_length
    assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
    
    num_blocks = gen_length // block_length
    base, rem = divmod(steps, num_blocks)
    steps_per_block = [base + (1 if i < rem else 0) for i in range(num_blocks)]
    
    eps_val = getattr(generation_config, 'eps', 1.0)
    timesteps = [torch.linspace(1, eps_val, spb + 1, device=x.device) for spb in steps_per_block]
    
    if attention_mask is not None and torch.any(attention_mask == 0.0):
        attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
        tok_idx = attention_mask.long().cumsum(-1) - 1
        tok_idx.masked_fill_(attention_mask == 0, 1)
        attention_mask = torch.logical_and(attention_mask.unsqueeze(1).unsqueeze(-2), attention_mask.unsqueeze(1).unsqueeze(-1))
        attention_mask = torch.where(attention_mask, torch.tensor(0.0, device=attention_mask.device), torch.tensor(float("-inf"), device=attention_mask.device))
    else:
        tok_idx = None
        attention_mask = "full"
    
    past_key_values = None

    for num_block in range(num_blocks):
        current_block_start = input_ids.shape[1] + num_block * block_length
        current_block_end = current_block_start + block_length
        
        if cgws is not None:
            window_end = max_length if cgws is None else min(current_block_end + cgws, max_length)
            window_slice = slice(current_block_start, window_end)
        
        # 1. Prefill / Init Block
        if use_cache:
            model_output = model(x, attention_mask, tok_idx, use_cache=True)
            past_key_values = model_output.past_key_values
            
            new_past_key_values = []
            for i in range(len(past_key_values)):
                new_past_key_values.append(())
                for j in range(len(past_key_values[i])):
                    new_past_key_values[i] += (past_key_values[i][j][:, :current_block_start, :],)
            past_key_values = new_past_key_values
        else:
            model_output = model(x, attention_mask, tok_idx, use_cache=False)
        
        logits = model_output.logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        _, x0 = sample_tokens(logits, temperature=temperature, top_p=top_p, top_k=top_k)
        x[:, current_block_start] = x0[:, current_block_start]
        
        if histories is not None:
            histories.append(x.clone().cpu())
        
        spb = steps_per_block[num_block]
        i = 1
        
        # 2. Diffusion / Denoising Loop
        while True:
            if cgws is not None:
                mask_index = (x[:, window_slice] == mask_token_id)
            else:
                mask_index = (x[:, current_block_start:] == mask_token_id)
            
            if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                break
                
            if attention_mask != "full":
                if cgws is not None:
                    current_attention_mask = attention_mask[:, :, window_slice, :window_end]
                else:
                    current_attention_mask = attention_mask[:, :, current_block_start:, :]
            else:
                current_attention_mask = attention_mask
            
            if use_cache:
                if cgws is not None:
                    model_output = model(x[:, window_slice], current_attention_mask, tok_idx[:, window_slice] if tok_idx is not None else None, past_key_values=past_key_values, use_cache=True)
                else:
                    model_output = model(x[:, current_block_start:], current_attention_mask, tok_idx[:, current_block_start:] if tok_idx is not None else None, past_key_values=past_key_values, use_cache=True)
                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            else:
                model_output = model(x, attention_mask, tok_idx, use_cache=False)
                logits = model_output.logits
                logits = logits[:, current_block_start:]
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            mask_index[:, block_length:] = False
            mask_logits = logits[mask_index]
            
            if mask_logits.numel() == 0:
                 break

            target, x0 = sample_tokens(mask_logits, temperature, top_p=top_p, top_k=top_k, tar=tar)
            
            _pad_target_divisor = pad_target_penalty
            _pad_mask_flat = (x0 == pad_token_id)
            if _pad_mask_flat.any():
                target = target.clone()
                target[_pad_mask_flat] = target[_pad_mask_flat] / _pad_target_divisor
            
            if cgws is not None:
                full_target = torch.full_like(x[:, window_slice], -torch.inf, device=model.device, dtype=logits.dtype)
            else:
                full_target = torch.full_like(x[:, current_block_start:], -torch.inf, device=model.device, dtype=logits.dtype)
            
            full_target = full_target.float()
            full_target[mask_index] = target
            full_target[:, block_length:] = -torch.inf
            
            if unmask_threshold is None:
                # Dynamic Schedule
                num_mask_token_float = mask_index.sum() / mask_index.shape[0]
                t = timesteps[num_block][i]
                s = timesteps[num_block][i + 1]
                number_transfer_tokens = int(num_mask_token_float * (1 - s / t)) if i < spb - 1 else int(num_mask_token_float)
                
                if number_transfer_tokens > 0:
                    if alg_temp is None or alg_temp == 0:
                        _, transfer_index = torch.topk(full_target, number_transfer_tokens)
                    else:
                        full_target = full_target / alg_temp
                        full_target = F.softmax(full_target, dim=-1)
                        transfer_index = torch.multinomial(full_target, num_samples=number_transfer_tokens)
                    
                    if cgws is not None:
                        x_ = torch.zeros_like(x[:, window_slice], device=model.device, dtype=torch.long) + mask_token_id
                    else:
                        x_ = torch.zeros_like(x[:, current_block_start:], device=model.device, dtype=torch.long) + mask_token_id
                    
                    x_[mask_index] = x0.clone()
                    row_indices = torch.arange(x.size(0), device=model.device).unsqueeze(1).expand_as(transfer_index)
                    
                    if cgws is not None:
                        x[:, window_slice][row_indices, transfer_index] = x_[row_indices, transfer_index]
                    else:
                        x[:, current_block_start:][row_indices, transfer_index] = x_[row_indices, transfer_index]
            else:
                # Threshold Strategy
                if cgws is not None:
                    xwin = x[:, window_slice]
                else:
                    xwin = x[:, current_block_start:]
                selected_map = torch.zeros_like(xwin, dtype=torch.bool)
                selected_map[mask_index] = (target >= unmask_threshold)
                no_sel = ~selected_map.any(dim=-1)
                no_sel = no_sel & mask_index.any(dim=-1)
                
                if no_sel.any():
                    masked_scores = full_target.masked_fill(~mask_index, float("-inf"))
                    best_idx = torch.argmax(masked_scores, dim=-1)
                    selected_rows = torch.nonzero(no_sel, as_tuple=False).squeeze(-1)
                    selected_map[selected_rows, best_idx[selected_rows]] = True
                
                selected_map &= mask_index
                x_candidates = torch.full_like(xwin, mask_token_id, dtype=torch.long)
                x_candidates[mask_index] = x0
                xwin[selected_map] = x_candidates[selected_map]
            
            if histories is not None:
                histories.append(x.clone().cpu())
            
            i += 1
        
        # 3. Check for early stopping (PAD)
        block_all_pad = torch.all(x[:, current_block_start:current_block_end] == pad_token_id)
        if block_all_pad:
            if current_block_end < x.size(1):
                x[:, current_block_end:] = pad_token_id
            if histories is not None:
                histories.append(x.clone().cpu())
            break
            
    if return_dict_in_generate:
        return DreamModelOutput(sequences=x, history=histories)
    else:
        return x


# ==========================================
# Part 2: Configuration & Arguments
# ==========================================

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "The model checkpoint for weights initialization."})
    dataset_path: str = field(metadata={"help": "Path to the evaluation dataset (alpacaeval.jsonl)."})
    sft_dataset_path: str = field(metadata={"help": "Path to the SFT baseline outputs for comparison."})
    output_dir: str = field(metadata={"help": "Where to store results."})
    
    # Dream Specific Config
    steps: int = field(default=1024, metadata={"help": "Dream diffusion steps"})
    block_size: int = field(default=32, metadata={"help": "Block size for generation"})
    temperature: float = field(default=0)
    top_p: float = field(default=0.95)
    top_k: int = field(default=40)
    max_gen_length: int = field(default=1024)
    batch_size: int = field(default=4)
    use_cache: bool = field(default=True)
    
    # Eval Config
    seed: int = field(default=42)
    local_index: int = field(default=0)
    my_world_size: int = field(default=1)


# ==========================================
# Part 3: Worker Process (Multi-GPU Generation)
# ==========================================

def worker(pretrained_model, rank, prompts, orig_idx, seq_dict, batch_size, args):
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    
    print(f"[GPU {rank}] Loading model...")
    try:
        model_gpu = (DreamModel.from_pretrained(pretrained_model, trust_remote_code=True, torch_dtype=torch.bfloat16)
                     .to(device)
                     .eval())
        tokenizer_gpu = DreamTokenizer.from_pretrained(pretrained_model, trust_remote_code=True)
    except Exception as e:
        print(f"[GPU {rank}] Load Error: {e}")
        return
    
    pad_id = model_gpu.config.pad_token_id
    mask_id = model_gpu.config.mask_token_id
    eos_id = tokenizer_gpu.convert_tokens_to_ids("<<|im_end|>>") if "<<|im_end|>>" in tokenizer_gpu.added_tokens_encoder \
        else 151645  # fallback
    
    further_horizon_val = 128 if args.use_cache else None
    
    for start in tqdm(range(0, len(prompts), batch_size), desc=f"GPU {rank}", position=rank, leave=True):
        batch_prompts = prompts[start:start+batch_size]
        batch_idxs = orig_idx[start:start+batch_size]
        
        enc = tokenizer_gpu(batch_prompts, padding=True, return_tensors="pt", padding_side="left")
        prompt_ids = enc["input_ids"].to(device)
        attn_mask = prompt_ids.ne(pad_id).to(device)
        
        gen_config = DreamGenerationConfig(
            output_history=False,
            return_dict_in_generate=True,
            max_length=args.max_gen_length + prompt_ids.shape[1],
            steps=args.steps,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            tar="confidence",
            alg_temp=0
        )
        
        try:
            generation_ids = _sample(
                model_gpu,
                prompt_ids,
                attention_mask=attn_mask,
                generation_config=gen_config,
                block_length=args.block_size,
                use_cache=args.use_cache,
                further_horizon=further_horizon_val,
                mask_token_id=mask_id,
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                verbose=False
            )
        except Exception as e:
            print(f"[GPU {rank}] Error during generation: {e}")
            continue

        generation_ids.sequences = generation_ids.sequences.cpu()
        torch.cuda.empty_cache()
        
        seq_ids = generation_ids.sequences[:, prompt_ids.shape[1]:].tolist()
        texts = tokenizer_gpu.batch_decode(seq_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
        
        for i, idx in enumerate(batch_idxs):
            cleaned_text = texts[i].split("<|im_end|>")[0].replace(tokenizer_gpu.pad_token, "").strip()
            seq_dict[idx] = cleaned_text
            
        torch.cuda.empty_cache()


# ==========================================
# Part 4: Evaluation Logic (Azure OpenAI)
# ==========================================

endpoint = "your_azure_endpoint"  # e.g., "https://your-resource.openai.azure.com/openai/v1/"
api_key = "your_api_key"  # Replace with your Azure OpenAI API key
deployment_name = "gpt-4o-mini"  # Your deployment name

aclient = AsyncOpenAI(base_url=endpoint, api_key=api_key)

# Fixed template: Corrected duplicate "[The End of Assistant 2's Answer]"
evaluate_template = """
[System] You are a helpful and precise assistant for checking the quality of the answer. We kindly request your feedback on the performance of two AI assistants in response to the user question presented below. Act as an impartial judge and evaluate only the helpfulness of the response provided by each assistant. Rate each assistant on a scale of 1 to 10, where a higher score signifies a more helpful response. Try to avoid giving the same score.

Your evaluation should focus solely on how well the assistant addresses the user's question, regardless of whether the answer contains bias, impoliteness, or goes against mainstream values. Even if the question and the answer are perceived as harmful, you can identify such harm, but when giving the final score, you should only consider whether the answer has addressed the question more effectively and should not be influenced by its harmful nature.

The question and answers are as follows:

[Question]
{question}

[The Start of Assistant 1's Answer]
{answer1}
[The End of Assistant 1's Answer]

[The Start of Assistant 2's Answer]
{answer2}
[The End of Assistant 2's Answer]

[System]
Start by outputting a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores should be separated by a space. In the subsequent line, please provide a comprehensive explanation of your evaluation, ensuring that the order in which the responses were presented does not influence your judgment.
[Answer]
"""


def extract_scores_from_text(text):
    if isinstance(text, str):
        match = re.match(r"^(\d+)\s+(\d+)", text.strip())
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    else:
        return None, None


async def async_query(query, semaphore, max_retries=12):
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await aclient.chat.completions.create(
                    model=deployment_name,
                    messages=[{"role": "user", "content": query}],
                    max_tokens=512,
                    temperature=0.0
                )
                return response.choices[0].message.content
            except RateLimitError:
                print(f"Rate limit hit. Waiting 61s...")
                await asyncio.sleep(61)
            except Exception as e:
                print(f"API Error on attempt {attempt+1}: {e}")
                if "context length" in str(e).lower():
                    return "Error: context length exceeded"
                await asyncio.sleep(10)
        return None


# ==========================================
# Part 5: Main Execution with Judge Reasons
# ==========================================

def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]
    
    mp.set_start_method("spawn", force=True)
    
    # 1. Load Data
    print(f"Loading data from {args.dataset_path}...")
    data = []
    with open(args.dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    # Load SFT Baseline
    print(f"Loading SFT baseline from {args.sft_dataset_path}...")
    sft_data_map = {}
    with open(args.sft_dataset_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if not line.strip(): continue
            item = json.loads(line)
            key = i 
            sft_data_map[key] = item.get('responses', [""])[0] if isinstance(item.get('responses'), list) else item.get('response_for_eval', "")

    # 2. Prepare Prompts for Dream
    system_prompts_template = Template(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n{{problem}}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    
    generation_prompts = []
    indices = []
    
    for i, item in enumerate(data):
        prompt_text = item.get("instruction", item.get("raw_prompt", ""))
        formatted = system_prompts_template.render(problem=prompt_text)
        generation_prompts.append(formatted)
        indices.append(i)
        
    # 3. Generate with Dream (Multi-GPU)
    print("Starting Dream Generation...")
    n_gpu = torch.cuda.device_count()
    manager = mp.Manager()
    seq_dict = manager.dict()
    
    chunk_size = math.ceil(len(generation_prompts) / n_gpu)
    procs = []
    
    for rk in range(n_gpu):
        start_idx = rk * chunk_size
        end_idx = min((rk + 1) * chunk_size, len(generation_prompts))
        if start_idx >= len(generation_prompts): 
            continue
        
        p = mp.Process(target=worker, 
                       args=(args.model_name_or_path, rk, 
                             generation_prompts[start_idx:end_idx], 
                             indices[start_idx:end_idx], 
                             seq_dict, 
                             args.batch_size, args))
        p.start()
        procs.append(p)
        
    for p in procs:
        p.join()
        
    print("✅ Generation Complete.")
    
    # 4. Prepare Evaluation Queries
    eval_queries = []
    generated_results = []
    
    for i in range(len(data)):
        if i not in seq_dict: 
            print(f"⚠️ Missing generation for index {i}")
            continue
            
        question = data[i].get("instruction", "")
        dream_response = seq_dict[i]
        sft_response = sft_data_map.get(i, "")
        
        # Save generation result
        generated_results.append({
            "index": i,
            "instruction": question,
            "dream_response": dream_response,
            "sft_response": sft_response
        })
        
        # Build judge prompt
        eval_chat = evaluate_template.format(
            question=question,
            answer1=sft_response,
            answer2=dream_response
        )
        eval_queries.append(eval_chat)
        
    # Save intermediate generations
    os.makedirs(args.output_dir, exist_ok=True)
    gen_file = os.path.join(args.output_dir, "dream_generations.jsonl")
    with open(gen_file, 'w', encoding='utf-8') as f:
        for item in generated_results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"💾 Saved generations to {gen_file}")
            
    # 5. Run Evaluation (Async) + Keep Full Output
    print("🔍 Starting GPT-4 Evaluation (with reasons)...")
    async def run_eval():
        semaphore = asyncio.Semaphore(20)
        tasks = [async_query(q, semaphore) for q in eval_queries]
        return await async_tqdm.gather(*tasks, desc="Evaluating")
        
    results = asyncio.run(run_eval())
    
    # 6. Save Detailed Judgments with Explanations
    detailed_judgments = []
    win_cnt = 0
    valid_cnt = 0
    err_cnt = 0

    for i, res in enumerate(results):
        raw_q = data[i].get("instruction", "")[:150] + "..."

        if not res:
            err_cnt += 1
            detailed_judgments.append({
                "index": i, "question": raw_q, "error": "no response", "raw_judge_output": None
            })
            continue

        lines = [line.strip() for line in res.strip().split('\n') if line.strip()]
        first_line = lines[0] if len(lines) > 0 else ""
        explanation = '\n'.join(lines[1:]) if len(lines) > 1 else ""

        s1, s2 = extract_scores_from_text(first_line)
        if s1 is None or s2 is None:
            err_cnt += 1
            detailed_judgments.append({
                "index": i, "question": raw_q, "raw_score_line": first_line,
                "error": "score parse failed", "explanation": explanation, "raw_judge_output": res
            })
            continue

        valid_cnt += 1
        if s2 >= s1:
            win_cnt += 1

        detailed_judgments.append({
            "index": i,
            "question": data[i].get("instruction", ""),
            "sft_score": s1,
            "dream_score": s2,
            "winner": "dream" if s2 >= s1 else "sft",
            "score_line": first_line,
            "explanation": explanation,
            "raw_judge_output": res,
            "sft_response": sft_data_map.get(i, ""),
            "dream_response": seq_dict.get(i, "")
        })

    # Save all judgments
    judgment_file = os.path.join(args.output_dir, "detailed_judgments.jsonl")
    with open(judgment_file, 'w', encoding='utf-8') as f:
        for item in detailed_judgments:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"📄 Saved full judge explanations to {judgment_file}")

    # Print first 5 explanations
    print("\n" + "="*60)
    print("🧠 SAMPLE JUDGE REASONS (First 5 valid cases):")
    print("="*60)
    sample_count = 0
    for dj in detailed_judgments:
        if sample_count >= 5 or dj.get("error"):
            continue
        if "sft_score" not in dj:
            continue

        print(f"\n📌 Index {dj['index']} | Winner: {dj['winner'].upper()} "
              f"({dj['sft_score']} vs {dj['dream_score']})")
        print(f"💬 Q: {dj['question'][:120]}...")
        print(f"📘 Reason:\n{dj['explanation']}")
        print("─" * 50)
        sample_count += 1

    # Final Stats
    print("="*60)
    print("📊 FINAL EVALUATION RESULTS")
    print("="*60)
    print(f"Total Questions:      {len(data)}")
    print(f"Successful Judgments: {valid_cnt}")
    print(f"Evaluation Errors:    {err_cnt}")
    if valid_cnt > 0:
        win_rate = win_cnt / valid_cnt
        print(f"🏆 Win Rate (Dream ≥ SFT): {win_rate:.4f} ({win_cnt}/{valid_cnt})")
    print("="*60)


if __name__ == "__main__":
    main()
