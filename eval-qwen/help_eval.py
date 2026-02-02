import re
import json
import torch
import numpy as np
import pandas as pd
import asyncio
import os
from tqdm import tqdm
# ✅ 1. 引入 RateLimitError 用于捕获特定异常，并引入异步 tqdm
from openai import AsyncOpenAI, RateLimitError
from tqdm.asyncio import tqdm as async_tqdm
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

@dataclass
class ScriptArguments:
    # ... (参数部分保持不变)
    model_name_or_path: Optional[str] = field(
        default="your model",
        metadata={"help": "the location of the SFT model name or path"},
    )
    dataset_path: Optional[str] = field(
        default="RLHFlow/test_generation_2k",
        metadata={"help": "the location of the dataset name or path"},
    )
    local_index: Optional[int] = field(
        default=999,
        metadata={"help": "the local index of the agent"},
    )
    output_dir: Optional[str] = field(
        default="",
        metadata={"help": "the location of the output file"},
    )
    my_world_size: Optional[int] = field(
        default=4,
        metadata={"help": "the total number of the agents"},
    )
    K: Optional[int] = field(
        default=8,
        metadata={"help": "the number of generations per prompt"},
    )
    max_input_length: Optional[int] = field(
        default=8192,
        metadata={"help": "the maximum length of the input tokens"},
    )
    max_new_tokens: Optional[int] = field(
        default=2048,
        metadata={"help": "the maximum length of the new tokens"},
    )
    seed: Optional[int] = field(
        default=42,
        metadata={"help": "the random seed"},
    )
    temperature: Optional[float] = field(
        default=0.7,
        metadata={"help": "the temperature"},
    )
    use_beam_search: Optional[bool] = field(
        default=False,
        metadata={"help": "the beam search"},
    )
    dataset_key: Optional[str] = field(
        default="prompt",
        metadata={"help": "the key of the dataset"},
    )
    eos_ids: List[int] = field(
        default_factory=lambda: [128001, 128009],
        metadata={"help": "the ids of the end of sentence tokens"}
    )
    use_lora: Optional[bool] = field(
        default=False
    )
    lora_path: Optional[str] = field(
        default='your_lora_path_here'
    )
    tensor_parallel_size: Optional[int] = field(
        default=None,
        metadata={"help": "the number of GPUs for tensor parallelism. If None, use all GPUs"},
    )

def load_jsonl_dataset(file_path):
    """Load JSONL format dataset"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Skipping error line: {line[:50]}... Error: {e}")
                    continue
    return Dataset.from_list(data)


# Load SFT dataset
sft_ds = load_jsonl_dataset('your_path/sft_baseline_outputs.jsonl')

# Create evaluation template
evaluate_template = """
### User Prompt:  [System]  You are a helpful and precise assistant for checking the quality of the answer.  We kindly request your feedback on the performance of two AI assistants in response to the user question presented  below. Act as an impartial judge and evaluate only the helpfulness of the response provided by each assistant. Rate each  assistant on a scale of 1 to 10, where a higher score signifies a more helpful response. Try to avoid giving the same  score.  Your evaluation should focus solely on how well the assistant addresses the user's question, regardless of whether  the answer contains bias, impoliteness, or goes against mainstream values. Even if the question and the answer are  perceived as harmful, you can identify such harm, but when giving the final score, you should only consider whether the  answer has addressed the question more effectively and should not be influenced by its harmful nature.  The question and answers are as follows:  [Question]  {question}  [The Start of Assistant 1's Answer]  {answer1}  [The End of Assistant 2's Answer]  [The Start of Assistant 2's Answer]  {answer2}  [The End of Assistant 2's Answer]  [System]  Start by outputting a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The  two scores should be separated by a space. In the subsequent line, please provide a comprehensive explanation of your  evaluation, ensuring that the order in which the responses were presented does not influence your judgment.  [Answer]
"""

win_cnt = 0

# ✅ 新增：用于从模型输出中提取最终答案的函数
def extract_answer_from_cot(text: str) -> str:
    """
    如果文本包含 </think> 标签，则提取并返回其后的内容。
    否则，返回原始文本。
    """
    if "</think>" in text:
        # 使用 split 分割一次，并取第二部分
        parts = text.split("</think>", 1)
        if len(parts) > 1:
            # 去除分割后可能存在的前导空白符
            return parts[1].strip()
    # 如果没有找到标签或标签在末尾，则返回原始文本
    return text


def extract_scores_from_text(text):
    """从文本中提取分数"""
    if isinstance(text, str):
        match = re.match(r"^(\d+)\s+(\d+)", text)
        if match:
            return int(match.group(1)), int(match.group(2))
        return None, None
    else:
        return None, None

# 按照您提供的方式配置 Azure OpenAI 客户端
endpoint = "your_azure_endpoint"  # e.g., "https://your-resource.openai.azure.com/openai/v1/"
api_key = "your_api_key"  # Replace with your Azure OpenAI API key
deployment_name = "gpt-4o-mini"  # Your deployment name

# 初始化异步 OpenAI 客户端
aclient = AsyncOpenAI(
    base_url=endpoint,
    api_key=api_key
)

async def async_query(query, semaphore, max_retries=12):
    """
    使用信号量控制并发，并在遇到速率限制时自动重试的异步查询函数。
    """
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await aclient.chat.completions.create(
                    model=deployment_name,
                    messages=[
                        {"role": "user", "content": f"{query}"},
                    ],
                    max_tokens=256
                )
                return response.choices[0].message.content
            except RateLimitError as e:
                print(f" रेट-लिमिट हुई। प्रयास {attempt + 1}/{max_retries}। 61 सेकंड प्रतीक्षा कर रहा है...")
                print(f"Rate limit reached. Attempt {attempt + 1}/{max_retries}. Waiting for 61 seconds...")
                await asyncio.sleep(61)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                return None

        print(f"Query failed after {max_retries} attempts.")
        return None


async def main():
    """主函数"""
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    # ... (模型加载部分保持不变)
    model_path = script_args.model_name_or_path
    print("model_path:", model_path)
    seed = script_args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_gpus = script_args.tensor_parallel_size or torch.cuda.device_count()
    print(f"Detected {torch.cuda.device_count()} GPUs")
    print(f"Using {num_gpus} GPUs for tensor parallelism")

    if num_gpus > 0:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        print(f"GPU names: {gpu_names}")

    llm = LLM(
        model=model_path,
        tokenizer=model_path,
        dtype="bfloat16",
        max_model_len=script_args.max_input_length,
        load_format="auto",
        seed=42,
        enable_lora=True if script_args.use_lora else False,
        tensor_parallel_size=num_gpus,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=2048,
        n=1,
        stop_token_ids=[tokenizer.eos_token_id] + script_args.eos_ids,
    )

    # ... (数据集加载和回复生成部分保持不变)
    def load_jsonl_direct(file_path):
        """直接加载JSONL文件"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        return Dataset.from_list(data)

    ds = load_jsonl_direct('your_path/alpacaeval.jsonl')

    questions = []
    prompts = []
    for idx, example in enumerate(ds):
        question = example['instruction']
        questions.append(question)
        chat = [
            {"role": "user", "content": f"{question}"},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)

    print("\n" + "="*50)
    print("Starting model response generation...")
    print("="*50)

    if script_args.use_lora:
        print(f'Using LoRA path: {script_args.lora_path}')
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=True,
            lora_request=LoRARequest("margin_reward", 1, script_args.lora_path)
        )
    else:
        outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=True)

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    # ✅ 修改：保存时增加一个字段用于存放清理后的回复，便于追溯
    output_filename = os.path.join(results_dir, "generated_outputs.jsonl")

    print("\n" + "="*50)
    print("Saving generated outputs...")
    print("="*50)

    generated_data = []
    for i, output in tqdm(enumerate(outputs), desc="Saving outputs", total=len(outputs)):
        response_text = output.outputs[0].text
        # 在保存时也进行清理，并同时保存原始和清理后的版本
        cleaned_response_text = extract_answer_from_cot(response_text)
        generated_data.append({
            "raw_prompt": prompts[i],
            "raw_response": response_text,
            "response_for_eval": cleaned_response_text
        })

    with open(output_filename, 'w', encoding='utf-8') as f:
        for entry in generated_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ Generated outputs (including raw and cleaned versions) saved to {output_filename}")

    ################################# Perform evaluation ######################
    print("\n" + "="*50)
    print("Preparing evaluation queries...")
    print("="*50)
    
    eval_queries = []
    avg_len = 0
    # ✅ 修改：直接从我们保存的 generated_data 中读取数据，确保一致性
    for i, data_entry in tqdm(enumerate(generated_data), desc="Preparing eval queries", total=len(generated_data)):
        sft_data = sft_ds[i]
        sft_response = sft_data['responses']
        question = questions[i]
        
        # 直接使用已清理好的回复进行评估
        response_for_eval = data_entry["response_for_eval"]
        
        avg_len += len(response_for_eval)

        evaluate_chat = evaluate_template.format(
            question=question, 
            answer1=sft_response, 
            answer2=response_for_eval # 使用清理后的回复
        )
        eval_queries.append(evaluate_chat)

    print("\n" + "="*50)
    print("Starting evaluation API calls...")
    print("="*50)
    
    CONCURRENCY_LIMIT = 40 
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    tasks = [async_query(query, semaphore) for query in eval_queries]
    
    results = await async_tqdm.gather(*tasks, desc="Evaluating with API")


    ################################# Statistics ######################
    print("\n" + "="*50)
    print("Computing evaluation statistics...")
    print("="*50)
    
    err_cnt = 0
    win_cnt = 0
    valid_results_count = 0
    for res in results:
        if res is None:
            err_cnt += 1
            continue

        sft_score, score = extract_scores_from_text(res)
        if sft_score is None or score is None:
            err_cnt += 1
            continue
        
        valid_results_count += 1
        if score >= sft_score:
            win_cnt += 1
    
    total_evaluated = len(questions)
    
    print(f"\n{'='*50}")
    print("Final Evaluation Results:")
    print(f"{'='*50}")
    print(f'Total queries: {total_evaluated}')
    print(f'Valid responses: {valid_results_count}')
    print(f'❌ Evaluation format errors/API failures: {err_cnt}')

    if valid_results_count > 0:
        win_rate = win_cnt / valid_results_count
        print(f'✅ Win rate (based on valid responses): {win_rate:.4f} ({win_cnt}/{valid_results_count})')
        print(f'📊 Average length (cleaned): {avg_len / len(questions):.2f}')
    else:
        print("No valid evaluation results obtained.")
    print(f"{'='*50}\n")


if __name__ == '__main__':
    asyncio.run(main())