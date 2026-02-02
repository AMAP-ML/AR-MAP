# Dream 7B
[![Static Badge](https://img.shields.io/badge/📰-Blog-red)](https://hkunlp.github.io/blog/2025/dream/)
[![Static Badge](https://img.shields.io/badge/📰-Report-yellow)](https://arxiv.org/abs/2508.15487)
[![Static Badge](https://img.shields.io/badge/📰-Demo-green)](https://huggingface.co/spaces/multimodalart/Dream)
[![Static Badge](https://img.shields.io/badge/Hugging%20Face%20🤗-Dream%207B_Base-blue)
](https://huggingface.co/Dream-org/Dream-v0-Base-7B)
[![Static Badge](https://img.shields.io/badge/Hugging%20Face%20🤗-Dream%207B_Instruct-blue)](https://huggingface.co/Dream-org/Dream-v0-Instruct-7B)

Dream is a 7B diffusion large language model that achieves competitive performance comparable to leading autoregressive models with a similar size.


## News
- [2025-09-26]: We release the training code for Dream, and you can easily fine-tune on your own dataset. Additionally, the training and evaluation code for [Dream-Coder](https://github.com/DreamLM/Dream-Coder) and [DreamOn](https://github.com/DreamLM/DreamOn) are all available now.
- [2025-07-15]: We release [Dream-Coder](https://github.com/DreamLM/Dream-Coder) and [DreamOn](https://github.com/DreamLM/DreamOn):
   - Dream-Coder is a fully open 7B dLLM for code, delivering strong performance, trained exclusively on public data.  
   - DreamOn tackles the variable-length generation and infilling problem in dLLMs.
- [2025-06-04]: Dream-Instruct eval code is released.
- [2025-05-03]: Dream-Base eval code is released.
- [2025-04-05]: Dream checkpoints and inference code are released.
- [2025-04-02]: Dream blog is released.


## Installation
Our implementation of Dream is based on the [Huggingface `transformers`](https://github.com/huggingface/transformers) library. You should first install transformers by `pip install transformers==4.46.2` and `torch==2.5.1` as Dream uses the [SdpaAttention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html) built in torch. Other versions of transformers and torch are not been fully tested.

Run the model requires a GPU with at least 20GB memory. 

Thanks [Apolinário](https://github.com/apolinario) for providing the online demo at https://huggingface.co/spaces/multimodalart/Dream.

## Usage
We provide several demos to show the inference code of Dream. A simple implementation is:
```python
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Dream-org/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

messages = [
    {"role": "user", "content": "Please write a Python class that implements a PyTorch trainer capable of training a model on a toy dataset."}
]
inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=512,
    output_history=True,
    return_dict_in_generate=True,
    steps=512,
    temperature=0.2,
    top_p=0.95,
    alg="entropy",
    alg_temp=0.,
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])
```

### Gradio demo

First, install [Gradio](https://www.gradio.app) `pip install gradio`, and then you can directly run `python app.py`
<div style="display: flex; justify-content: center; flex-wrap: wrap;">
    <img src="./imgs/example_gradio.gif" style="width: 80%" />
</div>

## Parameters of `diffusion_generate()` 

 `model.diffusion_generate()` supports a subset of arguments in `model.generate()` and some diffusion-specific arguments:
- `input_ids`: The input token ids.
- `attention_mask`: The attention mask when performing batch inference.
- `max_new_tokens`: The maximum tokens to generate. Note that the context length (input+output) of Dream currently is 2048.
- `output_history`: Whether to return the output at each intermediate step.
- `return_dict_in_generate`: The output format, mostly set to True.
- `steps`: The diffusion timesteps. `max_new_tokens`/`steps` tokens will be generated at each step. Fewer steps yield faster but coarser results.
- `temperature`: The value used to module the next token probabilities. By default 0.0. The smaller the value, the more accurate the results (e.g., in math or coding). The larger the value, the more diverse the results (e.g., in general conversation). If you notice repeated results, you might consider increasing the temperature.
- `top_p`: If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation. By default None. Control the diversity of generation. 
- `top_k`: The number of highest probability vocabulary tokens to keep for top-k-filtering. By default None. Control the diversity of generation.
- `alg`: The remasking strategy in diffusion sampling, controlling the token generation order. Support one random strategy and three confidence-based strategies:
    - `origin`: Token will be generated in a purely random order from https://arxiv.org/abs/2107.03006. The default strategy. Note this may degrade performance in some tasks.
    - `maskgit_plus`: Token will be generated based on the top1 confidence from https://arxiv.org/abs/2202.04200. 
    - `topk_margin`: Token will be generated based on the margin confidence by taking `top1 - top2` from https://arxiv.org/abs/2502.06768. 
    - `entropy`: Token will be generated based on the entropy of each token distribution. 
- `alg_temp`: Add some randomness to `alg` when using confidence-based strategies. By default None. 
- `generation_logits_hook_func`: a hook that can be user-defined to control the logits at each intermediate step, e.g., do some guidance.
- `generation_tokens_hook_func`: a hook that can be user-defined to control the tokens at each intermediate step, e.g., print, infill, or other token control strategies. See `demo_token_control.py` for reference.


## Evaluation
The evaluation is based on [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), so you should first install it with:
```
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
``` 
Then, you can go to the `eval` directory and run the bash scripts.
```
cd eval

# this scripts contains likelihood-based tasks: mmlu arc_easy arc_challenge hellaswag piqa gpqa_main_n_shot winogrande race
bash eval_dream_gen_mc.sh

# this scripts contains generation tasks: humaneval gsm8k_cot mbpp minerva_math bbh
bash eval_dream_gen.sh

# this scripts contains planning tasks: countdown, sudoku, trip-planning, their data are under `data`
bash eval_dream_gen_planning.sh
```

## Fine-tuning
You can perform SFT training on a single node with 8 GPUs with:
```
bash examples/run_sft_tulu3.sh 8 output
```

## Citation
```
@article{ye2025dream,
  title={Dream 7B: Diffusion Large Language Models},
  author={Ye, Jiacheng and Xie, Zhihui and Zheng, Lin and Gao, Jiahui and Wu, Zirui and Jiang, Xin and Li, Zhenguo and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2508.15487},
  year={2025}
}
```
