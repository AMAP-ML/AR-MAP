# ================== 0. Environment Setup ==================
import os as _os
_os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
_cache_root = "/dev/shm/torch_cache"
_os.makedirs(_cache_root, exist_ok=True)
_os.environ["TORCH_EXTENSIONS_DIR"] = _os.path.join(_cache_root, "torch_extensions")
_os.environ["TRITON_CACHE_DIR"] = _os.path.join(_cache_root, "triton")
_os.environ["XDG_CACHE_HOME"] = _cache_root
_os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")

_os.environ.pop("NCCL_BLOCKING_WAIT", None)
_os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)

# ================== 1. Imports and Parameter Definitions ==================
import os
import sys
import re
import json
import random
import traceback
import queue
import time
import torch
import torch.multiprocessing as mp
from termcolor import cprint
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import List, Optional
from datasets import load_dataset
from transformers import HfArgumentParser

# --- IFEval path compatibility ---
lm_eval_path = os.path.join(os.path.dirname(__file__))
if lm_eval_path not in sys.path:
    sys.path.insert(0, lm_eval_path)

# --- NLTK check ---
try:
    import nltk
    nltk.data.find("tokenizers/punkt_tab")
except ImportError:
    cprint("Error: 'nltk' library not installed. Please run `pip install nltk` to install.", "red")
    sys.exit(1)
except LookupError:
    # Try automatic silent download
    try:
        import nltk
        nltk.download('punkt_tab', quiet=True)
        nltk.download('punkt', quiet=True)
    except:
        pass

from lm_eval.tasks.ifeval.utils import (
    InputExample,
    test_instruction_following_strict,
    test_instruction_following_loose,
)

@dataclass
class ScriptArguments:
    model_name_or_path: str = field(metadata={"help": "Path to the JetEngine model."})
    output_dir: str = field(default="results", metadata={"help": "Directory to save logs and temporary files."})
    
    # JetEngine & Sampling parameters
    tensor_parallel_size: int = field(default=8, metadata={"help": "Tensor parallel size."})
    temperature: float = field(default=0.7, metadata={"help": "Sampling temperature."})
    top_p: float = field(default=0.95, metadata={"help": "Sampling top_p."})
    max_new_tokens: int = field(default=512, metadata={"help": "Max new tokens to generate."})
    block_size: int = field(default=4, metadata={"help": "JetEngine block size."})
    denoising_steps: int = field(default=4, metadata={"help": "JetEngine denoising steps."})
    max_active: int = field(default=256, metadata={"help": "Max active requests for JetEngine."})
    stop_token_list: List[int] = field(default_factory=lambda: [151645, 151643])

# ================== 2. Helper functions ==================

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

def extract_answer_from_cot(text: str) -> str:
    """Extract content after </think> tag; return original text if tag is missing"""
    if "</think>" in text:
        parts = text.split("</think>", 1)
        if len(parts) > 1: return parts[1].strip()
    return text

# ================== 3. Worker logic ==================
def _llm_worker_run(args):
    (model_path, tp, block_size, sampling_kwargs, vis_ids,
     prompts_slice, indices_slice, enforce_eager, max_active, store_port) = args
    
    # Set current process CUDA visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, vis_ids))
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(store_port)
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    
    # Patch sitecustomize for distributed port
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
        # Force enable Eager Mode to prevent deadlock during graph capture
        llm = LLM(model_path, enforce_eager=enforce_eager, tensor_parallel_size=tp, mask_token_id=151669, block_length=block_size, gpu_memory_utilization=0.95)
        sp = SamplingParams(**sampling_kwargs)
        local_max_active = min(max_active, max(1, len(prompts_slice)))
        
        # Use tqdm to show subprocess progress
        outs = llm.generate_streaming(prompts_slice, sp, max_active=local_max_active, use_tqdm=True)
        for j, o in enumerate(outs):
            results.append((indices_slice[j], o["text"]))
            
    except BaseException as e:
        print(f"[Worker PID={os.getpid()}] Error: {e}", flush=True)
        traceback.print_exc()
    finally:
        # Attempt to shutdown; if unsuccessful, the main process will force kill
        if llm is not None and hasattr(llm, "shutdown"): 
            try: llm.shutdown()
            except: pass
    return results

def _llm_worker_entry(args, out_q):
    try:
        res = _llm_worker_run(args)
        out_q.put(("ok", res))
    except BaseException:
        out_q.put(("err", {"pid": os.getpid(), "traceback": traceback.format_exc()}))

# ================== 4. Main Execution Flow ==================
def main():
    # 1. Process start method setup
    try:
        if mp.get_start_method(allow_none=True) != "spawn":
            mp.set_start_method("spawn", force=True)
    except RuntimeError: pass

    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    os.makedirs(script_args.output_dir, exist_ok=True)

    # 2. Load IFEval data
    cprint("\n" + "="*50, "yellow")
    cprint("Loading IFEval dataset...", "yellow")
    cprint("="*50, "yellow")
    
    dataset = load_dataset("google/IFEval", split="train")
    
    processed_data = []
    prompts = []
    
    for ex in dataset:
        p_text = ex.get("prompt", "")
        processed_data.append({
            "key": ex.get("key", 0),
            "instruction_id_list": ex.get("instruction_id_list", []),
            "prompt": p_text,
            "kwargs": ex.get("kwargs", []),
        })
        # Convert to ChatML format
        chatml_prompt = f"<|im_start|>user\n{p_text}<|im_end|>\n<|im_start|>assistant\n"
        prompts.append(chatml_prompt)

    cprint(f">>> Loaded {len(prompts)} test data items", "green")

    # 3. Prepare JetEngine generation parameters
    cprint("\n" + "="*50, "yellow")
    cprint("Starting model response generation (using JetEngine Multi-Process)...", "yellow")
    cprint("="*50, "yellow")

    tp = script_args.tensor_parallel_size
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        device_ids = [int(x.strip()) for x in cvd.split(",") if x.strip()]
    else:
        import torch
        device_ids = list(range(torch.cuda.device_count()))
    
    gpu_num = len(device_ids)
    assert gpu_num >= tp, f"Visible GPUs ({gpu_num}) < tensor_parallel_size ({tp})."

    ngroups = max(1, gpu_num // max(1, tp))
    groups = [device_ids[i*tp : (i+1)*tp] for i in range(ngroups)]
    cprint(f">>> Using {gpu_num} GPUs, forming {ngroups} worker groups with TP size {tp}.", "green")

    original_indices = list(range(len(prompts)))
    prompt_chunks = [prompts[i::ngroups] for i in range(ngroups)]
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
    
    # [Key fix] Force enable Eager Mode to prevent Graph Capture deadlock
    enforce_eager = True 
    cprint(">>> [Fix] Enforce Eager Mode is ENABLED to prevent Graph Capture deadlocks.", "magenta")

    # 4. Start processes
    for g in range(ngroups):
        if not prompt_chunks[g]: continue
        worker_args = (script_args.model_name_or_path, tp, script_args.block_size, sampling_kwargs, groups[g],
                       prompt_chunks[g], index_chunks[g], enforce_eager, script_args.max_active, base_port + g)
        p = ctx.Process(target=_llm_worker_entry, args=(worker_args, out_q))
        p.start(); procs.append(p)

    # 5. Collect results
    results_needed, results_got, all_results = len(procs), 0, []
    pbar = tqdm(total=len(prompts), desc="Generating with JetEngine")

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
    
    pbar.close()

    # 6. [Key fix] Forceful cleanup logic
    cprint(">>> All results collected. Forcefully cleaning up workers to prevent deadlocks...", "yellow")
    for p in procs:
        if not p.is_alive():
            continue

        p.join(timeout=1)
        if p.is_alive():
            try:
                p.terminate() # Send SIGTERM
                p.join(timeout=1)
                if p.is_alive():
                    p.kill() # Send SIGKILL
            except Exception as e:
                print(f"Error killing worker {p.pid}: {e}")
    
    # Organize results
    output_map = {idx: text for idx, text in all_results}
    generated_responses = [output_map.get(i, "") for i in range(len(prompts))]
    cprint("✅ Generation complete.", "green")

    # 7. IFEval evaluation logic
    cprint("\n" + "="*50, "yellow")
    cprint("Starting IFEval evaluation...", "yellow")
    cprint("="*50, "yellow")
    
    prompt_level_strict_acc_list = []
    inst_level_strict_acc_list = []
    prompt_level_loose_acc_list = []
    inst_level_loose_acc_list = []
    
    results_file = os.path.join(script_args.output_dir, "ifeval_results.jsonl")
    
    with open(results_file, "w", encoding="utf-8") as f:
        for i, response_text in enumerate(tqdm(generated_responses, desc="Evaluating Rules")):
            # Clean CoT content
            cleaned_response = extract_answer_from_cot(response_text)
            doc = processed_data[i]
            
            inp = InputExample(
                key=doc["key"],
                instruction_id_list=doc["instruction_id_list"],
                prompt=doc["prompt"],
                kwargs=doc["kwargs"],
            )
            
            # Call IFEval core evaluation functions
            out_strict = test_instruction_following_strict(inp, cleaned_response)
            out_loose = test_instruction_following_loose(inp, cleaned_response)
            
            # Statistics
            prompt_level_strict_acc_list.append(1.0 if out_strict.follow_all_instructions else 0.0)
            inst_level_strict_acc_list.extend(out_strict.follow_instruction_list)
            prompt_level_loose_acc_list.append(1.0 if out_loose.follow_all_instructions else 0.0)
            inst_level_loose_acc_list.extend(out_loose.follow_instruction_list)
            
            result_entry = {
                "key": doc["key"],
                "prompt": doc["prompt"],
                "raw_response": response_text,
                "cleaned_response": cleaned_response,
                "metrics": {
                    "strict": 1.0 if out_strict.follow_all_instructions else 0.0,
                    "loose": 1.0 if out_loose.follow_all_instructions else 0.0
                }
            }
            f.write(json.dumps(result_entry, ensure_ascii=False) + "\n")
    
    # 8. Final Statistics
    total_evaluated = len(generated_responses)
    p_strict = sum(prompt_level_strict_acc_list) / len(prompt_level_strict_acc_list) if prompt_level_strict_acc_list else 0.0
    i_strict = sum(inst_level_strict_acc_list) / len(inst_level_strict_acc_list) if inst_level_strict_acc_list else 0.0
    p_loose = sum(prompt_level_loose_acc_list) / len(prompt_level_loose_acc_list) if prompt_level_loose_acc_list else 0.0
    i_loose = sum(inst_level_loose_acc_list) / len(inst_level_loose_acc_list) if inst_level_loose_acc_list else 0.0
    
    cprint(f"\n{'='*50}\nFinal IFEval results:\n{'='*50}", "cyan")
    print(f'Total Prompts: {total_evaluated}')
    print(f'Prompt Level Strict Acc: {p_strict:.4f}')
    print(f'Inst Level Strict Acc:   {i_strict:.4f}')
    print(f'Prompt Level Loose Acc:  {p_loose:.4f}')
    print(f'Inst Level Loose Acc:    {i_loose:.4f}')
    
    # Save Summary
    summary_file = os.path.join(script_args.output_dir, "ifeval_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump({
            "prompt_level_strict_acc": p_strict,
            "inst_level_strict_acc": i_strict,
            "prompt_level_loose_acc": p_loose,
            "inst_level_loose_acc": i_loose,
        }, f, indent=2)
        
    print(f'\n✅ Results saved to: {script_args.output_dir}')

if __name__ == "__main__":
    main()