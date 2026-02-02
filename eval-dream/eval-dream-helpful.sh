#!/bin/bash
#SBATCH --job-name=dream_eval
#SBATCH --output=jupyter_logs/dream_eval-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

module avail
module load slurm "nvhpc-hpcx-cuda12/23.11"

# === Configuration ===
# Dream 
model_path=  "your_path/base_model"


data_path="eval-dataset/alpacaeval.jsonl"

# Baseline roll out
sft_baseline="eval-dataset/alpaca-dream-base.jsonl"


output_dir="your_dir_here"
mkdir -p ${output_dir}

# === Dream Hyperparameters ===
steps=1024
block_size=32
batch_size=4

# === Execution ===

python dream-helpful.py \
    --model_name_or_path ${model_path} \
    --dataset_path ${data_path} \
    --sft_dataset_path ${sft_baseline} \
    --output_dir ${output_dir} \
    --steps ${steps} \
    --block_size ${block_size} \
    --batch_size ${batch_size} \
    --max_gen_length 1024 \
    --temperature 0.7 \
    --top_p 0.95 

