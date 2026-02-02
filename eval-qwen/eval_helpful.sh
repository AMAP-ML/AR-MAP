#!/bin/bash
#SBATCH --job-name=eval_helpful
#SBATCH --output=jupyter_logs/eval-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=48:00:00
#SBATCH --account=your_account_here

module avail
module load slurm "nvhpc-hpcx-cuda12/23.11"

# Configuration
base_model=your_path/base_model
data_path=your_path/alpacaeval.jsonl
my_world_size=2
output_dir=your_path/results
use_lora=false
lora_path=your_path/lora_adapter

# Run evaluation
CUDA_VISIBLE_DEVICES="0,1,2,3" python help_eval.py \
  --model_name_or_path ${base_model} \
  --dataset_path ${data_path} \
  --output_dir ${output_dir} \
  --local_index 0 \
  --my_world_size ${my_world_size} \
  --use_lora ${use_lora} \
  --lora_path ${lora_path}