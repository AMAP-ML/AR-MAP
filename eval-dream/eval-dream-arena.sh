#!/bin/bash
#SBATCH --job-name=dream_eval_batch
#SBATCH --output=jupyter_logs/dream_eval_batch-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8

module avail
module load slurm "nvhpc-hpcx-cuda12/23.11"

# === Configuration ===
# Dataset path (fixed)
data_path="eval-dataset/arena-hard-base.jsonl"
sft_baseline="eval-dataset/arena-dream-base.jsonl"
output_base_dir="your_dir_here"
mkdir -p "${output_base_dir}"

# Dream hyperparameters (shared across all models)
steps=512
block_size=32
batch_size=2
max_gen_length=512
temperature=0.0
top_p=0.95

# === Model List ===
# You can add any number of model paths here
models=(
   "your_path/base_model"

)

# === Batch Execution ===
for model_path in "${models[@]}"; do
    # Extract model name from path as subdirectory (e.g., Dream-TIES-merged-truthful-PARALLEL)
    model_name=$(basename "${model_path}")
    output_dir="${output_base_dir}/${model_name}"
    mkdir -p "${output_dir}"

    echo "========================================"
    echo "Evaluating model: ${model_name}"
    echo "Output directory: ${output_dir}"
    echo "========================================"

    python dream-helpful.py \
        --model_name_or_path "${model_path}" \
        --dataset_path "${data_path}" \
        --sft_dataset_path "${sft_baseline}" \
        --output_dir "${output_dir}" \
        --steps "${steps}" \
        --block_size "${block_size}" \
        --batch_size "${batch_size}" \
        --max_gen_length "${max_gen_length}" \
        --temperature "${temperature}" \
        --top_p "${top_p}"
done

echo "All models evaluated!"
