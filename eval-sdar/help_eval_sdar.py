# ================== 0. Environment Setup (from your latest script) ==================
import os as _os
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
_cache_root = "/dev/shm/torch_cache"
_os.makedirs(_cache_root, exist_ok=True)
_os.environ["TORCH_EXTENSIONS_DIR"] = _os.path.join(_cache_root, "torch_extensions")
_os.environ["TRITON_CACHE_DIR"] = _os.path.join(_cache_root, "triton")
_os.environ["XDG_CACHE_HOME"] = _cache_root
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
_os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
_os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
_os.environ.pop("NCCL_BLOCKING_WAIT", None)
_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

# ================== 1. Imports and Parameter Definitions ==================
import os
import sys
import re
import json
import random
import math
import socket
import traceback
import queue
import asyncio
import torch
import torch.multiprocessing as mp
from jinja2 import Template
from termcolor import cprint
from tqdm import tqdm
from openai import AsyncOpenAI, RateLimitError
from tqdm.asyncio import tqdm as async_tqdm
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, HfArgumentParser


@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the JetEngine model."})
    dataset_path: str = field(default="your_path/arena-hard.jsonl", metadata={"help": "Path to the arena-hard.jsonl input file."})
    output_dir: str = field(default="results", metadata={"help": "Directory to save logs and temporary files."})

    # JetEngine & Sampling Parameters
    tensor_parallel_size: int = field(default=2, metadata={"help": "Tensor parallel size."})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=0.95, metadata={"help": "Sampling top_p."})
    max_new_tokens: int = field(default=2048, metadata={"help": "Max new tokens to generate."})
    block_size: int = field(default=4, metadata={"help": "JetEngine block size."})
    denoising_steps: int = field(default=4, metadata={"help": "JetEngine denoising steps."})
    max_active: int = field(default=256, metadata={"help": "Max active requests for JetEngine."})
    stop_token_list: List[int] = field(default_factory=lambda: [151645, 151643])

    # Retained old parameters (may no longer be used, but for compatibility)
    seed: int = field(default=42)
    use_lora: bool = field(default=False) # JetEngine does not support LoRA
    lora_path: Optional[str] = field(default=None)

# ================== 2. Helper Functions (Merged from two scripts) ==================

# --- JetEngine Part ---
def _find_free_port():
    s = socket.socket(); s.bind(('', 0)); p = s.getsockname()[1]; s.close(); return p

def _patch_dist_port(port: int):
    import torch.distributed as _dist
    _real_init = _dist.init_process_group
    def _wrapped(backend, init_method=None, *args, **kwargs):
        if isinstance(init_method, str) and init_method.startswith("tcp://localhost:2333"):
            init_method = f"tcp://127.0.0.1:{port}"
        return _real_init(backend, init_method, *args, **kwargs)
    _dist.init_process_group = _wrapped

def _patch_safe_destroy():
    import torch.distributed as dist
    _real_destroy = dist.destroy_process_group
    def _safe_destroy(group=None):
        try:
            if not dist.is_available() or not dist.is_initialized(): return
            _real_destroy(group)
        except (AssertionError, Exception): pass
    dist.destroy_process_group = _safe_destroy

# --- Evaluation Part ---
def load_jsonl_dataset(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try: data.append(json.loads(line))
                except json.JSONDecodeError as e: print(f"Skipping error line: {line[:50]}... Error: {e}")
    return Dataset.from_list(data)

def extract_answer_from_cot(text: str) -> str:
    if "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) > 1: return parts[1].strip()
    return text

def extract_scores_from_text(text):
    if isinstance(text, str):
        match = re.match(r"^\s*(\d{1,2})\s+(\d{1,2})", text)
        if match: return int(match.group(1)), int(match.group(2))
    return None, None

# ================== 3. Worker Logic (from your latest script) ==================
def _llm_worker_run(args):
    (model_path, tp, block_size, sampling_kwargs, vis_ids,
     prompts_slice, indices_slice, enforce_eager, max_active, store_port) = args
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, vis_ids))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    patch_dir = f"/tmp/je_site_{store_port}"
    os.makedirs(patch_dir, exist_ok=True)
    with open(os.path.join(patch_dir, "sitecustomize.py"), "w") as f:
        f.write("import os\nimport torch.distributed as dist\n_real = dist.init_process_group\ndef _wrapped(backend, init_method=None, *args, **kwargs):\n    port = os.environ.get('JE_TCP_PORT')\n    if port and isinstance(init_method, str) and init_method.startswith('tcp://localhost:2333'):\n        init_method = f'tcp://127.0.0.1:{port}'\n    return _real(backend, init_method, *args, **kwargs)\ndist.init_process_group = _wrapped\n")
    os.environ["PYTHONPATH"] = patch_dir + (":" + os.environ.get("PYTHONPATH", ""))
    os.environ["JE_TCP_PORT"] = str(store_port)
    import torch
    _patch_dist_port(store_port)
    _patch_safe_destroy()
    torch.cuda.set_device(0)
    print(f"[Worker PID={os.getpid()}] CVD={os.environ['CUDA_VISIBLE_DEVICES']}, Port={store_port}, Prompts={len(prompts_slice)}", flush=True)
    from jetengine_ext.llm import LLM
    from jetengine_ext.sampling_params import SamplingParams
    llm, results = None, []
    try:
        llm = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp, mask_token_id=151669, block_length=block_size, gpu_memory_utilization=0.95)
        sp = SamplingParams(**sampling_kwargs)
        local_max_active = min(max_active, max(1, len(prompts_slice)))
        # Use use_tqdm=True here to display the progress bar of the subprocess
        outs = llm.generate_streaming(prompts_slice, sp, max_active=local_max_active, use_tqdm=True)
        for j, o in enumerate(outs):
            results.append((indices_slice[j], o["text"]))
    except BaseException as e:
        print(f"[Worker PID={os.getpid()}] Error: {e}", flush=True)
        traceback.print_exc()
    finally:
        if llm is not None and hasattr(llm, "shutdown"): llm.shutdown()
    return results

def _llm_worker_entry(args, out_q):
    try:
        res = _llm_worker_run(args)
        out_q.put(("ok", res))
    except BaseException:
        out_q.put(("err", {"pid": os.getpid(), "traceback": traceback.format_exc()}))

# ================== 4. Main `async` Evaluation Function ==================
async def main():
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    
    # ✅ **New**: Ensure output directory exists
    os.makedirs(script_args.output_dir, exist_ok=True)

    # --- Set multi-GPU environment variables ---
    tp = script_args.tensor_parallel_size
    if tp == 1:
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
        os.environ.setdefault("NCCL_IB_DISABLE", "1")
    else:
        for k in ["NCCL_P2P_DISABLE", "NCCL_IB_DISABLE"]: os.environ.pop(k, None)

    # --- Data Loading ---
    sft_ds = load_jsonl_dataset('your_path/sft_baseline_outputs.jsonl')
    eval_ds = load_jsonl_dataset(script_args.dataset_path)
    questions = [ex['instruction'] for ex in eval_ds]
    
    # --- Prompt Preparation ---
    prompt_tmpl = Template("<|im_start|>user\n{{problem}}<|im_end|>\n<|im_start|>assistant\n")
    all_prompts = [prompt_tmpl.render(problem=q) for q in questions]

    # #################################################################
    # Replacement part: Generate using JetEngine
    # #################################################################
    cprint("\n" + "="*50, "yellow")
    cprint("Start generating model responses (using JetEngine Multi-Process)...", "yellow")
    cprint("="*50, "yellow")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    device_ids = [int(x.strip()) for x in cvd.split(",") if x.strip()] if cvd else list(range(torch.cuda.device_count()))
    gpu_num = len(device_ids)
    assert gpu_num >= tp, f"Visible GPUs ({gpu_num}) < tensor_parallel_size ({tp})."
    
    ngroups = max(1, gpu_num // max(1, tp))
    groups = [device_ids[i*tp : (i+1)*tp] for i in range(ngroups)]
    cprint(f">>> Using {gpu_num} GPUs, forming {ngroups} worker groups with TP size {tp}.", "green")

    original_indices = list(range(len(all_prompts)))
    prompt_chunks = [all_prompts[i::ngroups] for i in range(ngroups)]
    index_chunks = [original_indices[i::ngroups] for i in range(ngroups)]
    
    sampling_kwargs = {
        "temperature": script_args.temperature, "topk": 0, "topp": script_args.top_p,
        "max_tokens": script_args.max_new_tokens, "stop_words": script_args.stop_token_list,
        "remasking_strategy": "low_confidence_dynamic", "dynamic_threshold": 0.9,
        "denoising_steps": script_args.denoising_steps, "block_length": script_args.block_size,
    }

    ctx = mp.get_context("spawn")
    out_q = ctx.Queue()
    procs = []
    base_port = 29000 + random.randint(0, 100)
    enforce_eager = tp <= 1
    
    for g in range(ngroups):
        if not prompt_chunks[g]: continue
        worker_args = (script_args.model_name_or_path, tp, script_args.block_size, sampling_kwargs, groups[g],
                       prompt_chunks[g], index_chunks[g], enforce_eager, script_args.max_active, base_port + g)
        p = ctx.Process(target=_llm_worker_entry, args=(worker_args, out_q))
        p.start(); procs.append(p)
    
    results_needed, results_got, all_results = len(procs), 0, []
    pbar = tqdm(total=len(all_prompts), desc="Generating with JetEngine")
    
    while results_got < results_needed:
        try:
            kind, payload = out_q.get(timeout=3600)
            if kind == "ok":
                all_results.extend(payload)
                results_got += 1
                pbar.update(len(payload))
                cprint(f"\n>>> Worker finished. {results_got}/{results_needed} complete.", "magenta")
            else:
                cprint(f"\nFATAL: Worker error:\n{payload['traceback']}", "red"); [p.terminate() for p in procs if p.is_alive()]; sys.exit(1)
        except queue.Empty:
            dead = [p for p in procs if not p.is_alive()]
            if dead: cprint(f"\nFATAL: {len(dead)} worker(s) died unexpectedly.", "red"); sys.exit(1)
            continue
    for p in procs: p.join()
    pbar.close()

    output_map = {idx: text for idx, text in all_results}
    generated_responses = [output_map.get(i, "") for i in range(len(all_prompts))]
    cprint("✅ Generation complete.", "green")

    # #################################################################
    # Replacement end: Now `generated_responses` contains all generated text
    # #################################################################

    # Prepare evaluation queries
    cprint("\n" + "="*50, "yellow"); cprint("Preparing eval queries...", "yellow"); cprint("="*50, "yellow")
    eval_queries = []
    avg_len = 0
    total_evaluated = len(questions)

    for i in tqdm(range(total_evaluated), desc="Preparing eval queries"):
        response_text = generated_responses[i]
        cleaned_response_text = extract_answer_from_cot(response_text)
        sft_response = sft_ds[i]['responses']
        avg_len += len(cleaned_response_text)
        
        evaluate_template = """### User Prompt:  [System]  You are a helpful and precise assistant for checking the quality of the answer.  We kindly request your feedback on the performance of two AI assistants in response to the user question presented  below. Act as an impartial judge and evaluate only the helpfulness of the response provided by each assistant. Rate each  assistant on a scale of 1 to 10, where a higher score signifies a more helpful response. Try to avoid giving the same  score.  Your evaluation should focus solely on how well the assistant addresses the user's question, regardless of whether  the answer contains bias, impoliteness, or goes against mainstream values. Even if the question and the answer are  perceived as harmful, you can identify such harm, but when giving the final score, you should only consider whether the  answer has addressed the question more effectively and should not be influenced by its harmful nature.  The question and answers are as follows:  [Question]  {question}  [The Start of Assistant 1's Answer]  {answer1}  [The End of Assistant 1's Answer]  [The Start of Assistant 2's Answer]  {answer2}  [The End of Assistant 2's Answer]  [System]  Start by outputting a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The  two scores should be separated by a space. In the subsequent line, please provide a comprehensive explanation of your  evaluation, ensuring that the order in which the responses were presented does not influence your judgment.  [Answer]"""
        evaluate_chat = evaluate_template.format(question=questions[i], answer1=sft_response, answer2=cleaned_response_text)
        eval_queries.append(evaluate_chat)

    # Call API
    cprint("\n" + "="*50, "yellow"); cprint("Starting API calls for evaluation...", "yellow"); cprint("="*50, "yellow")
    endpoint = "your_azure_endpoint"  # e.g., "https://your-resource.openai.azure.com/openai/v1/"
    api_key = "your_api_key"  # Replace with your Azure OpenAI API key
    deployment_name = "gpt-4o-mini"
    aclient = AsyncOpenAI(base_url=endpoint, api_key=api_key)
    
    async def async_query(query, semaphore):
        async with semaphore:
            for attempt in range(12):
                try:
                    response = await aclient.chat.completions.create(model=deployment_name, messages=[{"role": "user", "content": f"{query}"}], max_tokens=400, temperature=0.0)
                    return response.choices[0].message.content
                except RateLimitError: await asyncio.sleep(61)
                except Exception as e: print(f"Error during API call: {e}"); return None
            return None
            
    semaphore = asyncio.Semaphore(40)
    tasks = [async_query(query, semaphore) for query in eval_queries]
    api_results = await async_tqdm.gather(*tasks, desc="Evaluating with API")

    # ===================== ✅ Modification Start: Tally results and save/print details =====================
    cprint("\n" + "="*50, "yellow"); cprint("Tallying results and saving detailed info...", "yellow"); cprint("="*50, "yellow")
    
    err_cnt, win_cnt = 0, 0
    evaluation_details = []

    for i, res in enumerate(api_results):
        cleaned_model_response = extract_answer_from_cot(generated_responses[i])
        sft_response = sft_ds[i]['responses']

        detail_record = {
            "index": i,
            "question": questions[i],
            "baseline_response(sft)": sft_response,
            "model_response": cleaned_model_response,
            "judge_output": res,
            "score_baseline": None,
            "score_model": None,
            "error": None
        }

        if res is None:
            err_cnt += 1
            detail_record["error"] = "API call failed or returned None"
        else:
            sft_score, score = extract_scores_from_text(res)
            if sft_score is None or score is None:
                err_cnt += 1
                detail_record["error"] = "Failed to parse scores from judge output"
            else:
                detail_record["score_baseline"] = sft_score
                detail_record["score_model"] = score
                if score >= sft_score:
                    win_cnt += 1
        
        evaluation_details.append(detail_record)

        # ✅ **New**: Print a valid judge result every 20 items
        if (i + 1) % 20 == 0 and res is not None:
            cprint(f"\n--- [ Judge Result {i+1}/{total_evaluated} ] ---", "blue")
            print(res)
            cprint("------------------------------------", "blue")

    # ✅ **New**: Save detailed evaluation results to a file
    detailed_results_filename = os.path.join(script_args.output_dir, "evaluation_details.jsonl")
    with open(detailed_results_filename, 'w', encoding='utf-8') as f:
        for record in evaluation_details:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
    
    cprint(f"\n✅ Detailed evaluation results saved to {detailed_results_filename}", "green")

    # ===================== ✅ Modification End =====================

    valid_results_count = total_evaluated - err_cnt
    
    cprint(f"\n{'='*50}\nFinal Evaluation Statistics:\n{'='*50}", "cyan")
    print(f'Total queries: {total_evaluated}')
    print(f'Valid responses: {valid_results_count}')
    print(f'❌ Format errors/API failures: {err_cnt}')
    if valid_results_count > 0:
        win_rate = win_cnt / valid_results_count
        print(f'✅ Win rate (based on valid responses): {win_rate:.4f} ({win_cnt}/{valid_results_count})')
        print(f'📊 Average length (after cleaning): {avg_len / total_evaluated:.2f}')
    else:
        print("No valid evaluation results obtained.")
    print(f"{'='*50}\n")

if __name__ == '__main__':
    asyncio.run(main())