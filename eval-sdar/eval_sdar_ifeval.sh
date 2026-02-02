#!/bin/bash
#SBATCH --job-name=eval_ifeval
#SBATCH --output=jupyter_logs/eval-ifeval-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --account=your_account_here

# --- CONFIGURATION ---
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# --- Output base directory (each model will have its own subfolder)
base_output_dir=your_path/results

# --- JetEngine parameters (same for all models)
tensor_parallel_size=8
max_active=128
block_size=4
denoising_steps=4

# --- LIST OF MODELS TO EVALUATE ---
models=(
    "your_path/model1"
    "your_path/model2"
    # Add more models as needed
)

# --- LOOP OVER MODELS ---
for model_path in "${models[@]}"; do
    echo "=========================================="
    echo "Evaluating model: ${model_path}"
    echo "=========================================="
    
    model_name=$(basename "${model_path}")
    output_dir="${base_output_dir}/${model_name}"
    
    python ifeval_eval_sdar.py \
        --model_name_or_path "${model_path}" \
        --output_dir "${output_dir}" \
        --tensor_parallel_size ${tensor_parallel_size} \
        --max_active ${max_active} \
        --block_size ${block_size} \
        --denoising_steps ${denoising_steps}
    
    echo "Finished evaluating: ${model_name}"
    echo ""
done

echo "All evaluations complete!"
for model_path in "${models[@]}"; do
    echo "========================================"
    echo "Evaluating model: $model_path"
    echo "Start time: $(date)"
    echo "========================================"

    # Extract a clean name for the output folder (e.g., last part of path)
    model_name=$(basename "$model_path")
    output_dir="${base_output_dir}/${model_name}"

    # Ensure output directory exists
    mkdir -p "$output_dir"

    # Run evaluation
    python ifeval_eval_sdar.py \
        --model_name_or_path "$model_path" \
        --output_dir "$output_dir" \
        --tensor_parallel_size "$tensor_parallel_size" \
        --max_active "$max_active" \
        --block_size "$block_size" \
        --denoising_steps "$denoising_steps"

    echo "Finished evaluating: $model_path at $(date)"
    echo ""
done

echo "All models evaluated!"
