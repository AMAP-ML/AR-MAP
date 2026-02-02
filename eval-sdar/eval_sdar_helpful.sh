#!/bin/bash
#SBATCH --job-name=eval_sdar_helpful
#SBATCH --output=jupyter_logs/eval-jet-final-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --account=your_account_here

# --- CONFIGURATION ---
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# --- Model & Data Paths ---
model_path=your_path/model
data_path=your_path/alpacaeval.jsonl
output_dir=your_path/results

# --- JetEngine Specific Parameters ---
tensor_parallel_size=1
max_active=256
block_size=4
denoising_steps=4

# --- EXECUTION ---
python help_eval_sdar.py \
    --model_name_or_path ${model_path} \
    --dataset_path ${data_path} \
    --output_dir ${output_dir} \
    --tensor_parallel_size ${tensor_parallel_size} \
    --max_active ${max_active} \
    --block_size ${block_size} \
    --denoising_steps ${denoising_steps}