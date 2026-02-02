<div align="center">
  <h1>AR-MAP</h1>
  <h3>Are Autoregressive Large Language Models Implicit Teachers for Diffusion Large Language Models?</h3>
  <h4>A comprehensive framework for transferring alignment knowledge from AR-LLMs to Diffusion Models</h4>
</div>

<p align="center">
  <a href="AR-MAP/AR-MAP.pdf">
    <img
      src="https://img.shields.io/badge/Paper-PDF-red?logo=adobe&logoColor=red"
      alt="AR-MAP Paper"
    />
  </a>
  <a href="https://huggingface.co/aijwhedqie/AR-MAP-weight/tree/main">
    <img
      src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow"
      alt="Hugging Face Models"
    />
  </a>
  <a href="#quick-start">
    <img 
        src="https://img.shields.io/badge/Quick-Start-green?logo=rocket" 
        alt="Quick Start" 
    />
  </a>
  <a href="#features">
    <img 
        src="https://img.shields.io/badge/Features-Overview-blue?logo=star" 
        alt="Features"
    />
  </a>
</p>

<p align="center">
  <img src="assets/framework.png" alt="AR-MAP Framework Cover" width="800">
  <br>
  <em>Figure: The AR-MAP Framework. Transferring alignment from AR Teachers to Diffusion Students.</em>
</p>

## 📖 Overview

**AR-MAP** (Autoregressive Model Alignment for Diffusion) is a novel transfer learning framework that leverages preference-aligned Autoregressive LLMs (AR-LLMs) as implicit teachers for Diffusion LLMs (DLLMs). This repository contains the complete implementation including:

- **Multi-aspect DPO training** for helpfulness, truthfulness, and mathematical reasoning
- **Comprehensive evaluation suite** across multiple benchmarks
- **Model merging utilities** for LoRA adapters
- **Support for multiple model architectures** (Qwen, Dream, SDAR)

## 🌟 Features

- **Multi-Aspect Optimization**: Train models on multiple preference dimensions simultaneously
  - Helpfulness alignment
  - Truthfulness enhancement  
  - Mathematical reasoning improvement
  
- **Flexible Training Pipeline**: 
  - DPO (Direct Preference Optimization) training
  - LoRA fine-tuning support
  - Multi-GPU distributed training
  
- **Comprehensive Evaluation**:
  - AlpacaEval for helpfulness
  - TruthfulQA for truthfulness
  - Arena-Hard for general capabilities
  - Automated GPT-4 based evaluation

- **Model Support**:
  - Qwen 2.5 series
  - Dream diffusion models
  - SDAR models
  - Easy extension to other architectures

## 🚀 Quick Start

### Installation

```bash
conda create --name armap python=3.10
conda activate armap
pip install torch==2.6.0
pip install --no-cache-dir \
  https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/\
flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install -r requirements.txt
```

### Basic Usage

#### 1. Training with DPO

Use LlamaFactory for DPO training:

```bash
cd LlamaFactory-main
```

#### 2. Merging LoRA Adapters

After training, merge LoRA weights back to the base model:

```bash
# For Qwen models
python merge-lora-ar.py \
  --base_model your_path/base_model \
  --lora_adapter your_path/lora_adapter \
  --output your_path/merged_model \
  --weight 1.0

# For Dream models
python merge-lora-dream.py \
  --base_model your_path/Dream-base \
  --lora_adapter your_path/lora_adapter \
  --output your_path/merged_model \
  --weight 6.0

# For SDAR models
python merge-lora-sdar.py \
  --base_model your_path/SDAR-base \
  --lora_adapter your_path/lora_adapter \
  --output your_path/merged_model \
  --weight 6.0
```

#### 3. Evaluation

Evaluate your models on various benchmarks:

```bash
# Helpfulness evaluation (AlpacaEval)
cd eval-qwen
bash eval_helpful.sh

# Truthfulness evaluation
cd eval-qwen
python help_eval.py --model_name_or_path your_path/model

# Arena-Hard evaluation
cd eval-qwen
bash eval_arena.sh
```

## 📁 Project Structure

```
AR-MAP/
├── merge-lora-ar.py          # LoRA merging for Qwen models
├── merge-lora-dream.py        # LoRA merging for Dream models
├── merge-lora-sdar.py         # LoRA merging for SDAR models
├── eval-qwen/                 # Evaluation scripts for Qwen
│   ├── help_eval.py          # Helpfulness evaluation
│   ├── arena_qwen3.py        # Arena-Hard evaluation
│   └── eval_*.sh             # Evaluation bash scripts
├── eval-dream/                # Evaluation scripts for Dream
│   ├── dream-helpful.py      # Helpfulness evaluation
│   ├── dream-truthful.py     # Truthfulness evaluation
│   └── dream/                # Dream model implementation
├── eval-sdar/                 # Evaluation scripts for SDAR
│   ├── help_eval_sdar.py     # Helpfulness evaluation
│   ├── sdar_truthful.py      # Truthfulness evaluation
│   ├── ifeval_eval_sdar.py   # IFEval benchmark
│   └── jetengine_ext/        # Optimized inference engine
├── eval-dataset/              # Evaluation datasets
│   ├── alpaca-*.jsonl        # AlpacaEval datasets
│   ├── arena-*.jsonl         # Arena-Hard datasets
│   └── TruthfulQA.csv        # TruthfulQA dataset
├── train-dataset/             # Training datasets
│   ├── dpo_helpful.json      # Helpfulness preference data
│   ├── dpo_math.json         # Math preference data
│   └── dpo_truthful.json     # Truthfulness preference data

├── LlamaFactory-main/         # Training framework
└── requirements.txt           # Python dependencies
```

## 🔧 Configuration

### Model Paths

Update the following paths in the scripts to match your setup:

```python
# In merge-lora-*.py
BASE_MODEL_PATH = "your_path/base_model"
LORA_PATH = "your_path/lora_adapter"
OUTPUT_PATH = "your_path/merged_model"

# In eval scripts
model_name_or_path = "your_path/model"
dataset_path = "your_path/dataset"
```

### API Configuration

For GPT-4 based evaluation, configure your API endpoint:

```python
# In evaluation scripts
endpoint = "your_api_endpoint"
api_key = "your_api_key"  # Keep this secure!
deployment_name = "your_deployment"
```

**Note**: Remove sensitive API keys before sharing code publicly.

## 📊 Evaluation Metrics

Our framework evaluates models across multiple dimensions:

- **Helpfulness**: Measured via AlpacaEval with GPT-4 as judge
- **Truthfulness**: Evaluated on TruthfulQA benchmark
- **Mathematical Reasoning**: Tested on MATH and GSM8K datasets, Please note that we use the framework in [TraceRL](https://github.com/Gen-Verse/dLLM-RL) for evaluation.
- **General Capabilities**: Arena-Hard benchmark
- **Instruction Following**: IFEval benchmark

## 🎯 Training Data

The training datasets are organized by aspect:

- `dpo_helpful.json`: Preference pairs for helpfulness
- `dpo_math.json`: Preference pairs for mathematical reasoning
- `dpo_truthful.json`: Preference pairs for truthfulness

Each dataset contains pairs of (chosen, rejected) responses for DPO training.

## 🔬 Model Architectures

### Supported Models

1. **QwenSeries**: Standard autoregressive models
2. **Dream**: Diffusion-based language models with block attention
3. **SDAR**: Semi-autoregressive diffusion models

### Merging Strategies

Different models require different merging coefficients:

- **Qwen**: Standard merging (weight=1.0)
- **Dream/SDAR**: Higher coefficients (weight=3.0) for better performance

## 📈 Results

Please refer to our paper (`ARMAP_ARXIV.pdf`) for detailed experimental results and analysis.

## 🛠️ Advanced Usage





## 🤝 Acknowledgements

This work builds upon several excellent open-source projects:

- [LlamaFactory](https://github.com/hiyouga/LLaMA-Factory) for training infrastructure
- [Dream](https://github.com/DreamLM/Dream) for diffusion language models
- [SDAR](https://github.com/JetAstra/SDAR) for semi-autoregressive models
- [TraceRL](https://github.com/Gen-Verse/dLLM-RL) for evaluation framework

## 📝 Citation

If you find this work useful, please cite our paper.



## 📄 License

This project is released under the MIT License. See LICENSE file for details.

## 🔗 Contact

For questions or issues, please open an issue on GitHub or contact the authors.

---

<div align="center">
  Made with ❤️ for better language model alignment
</div>
