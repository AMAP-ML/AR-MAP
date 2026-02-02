#!/bin/bash
#SBATCH --job-name=eval_arena_batch
#SBATCH --output=jupyter_logs/eval-arena-batch-%J.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --time=48:00:00
#SBATCH --account=your_account_here

# ======================= CONFIGURATION =======================
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"

# --- Paths and Parameters ---
DATA_PATH=your_path/arena-hard.jsonl
MAIN_OUTPUT_DIR=your_path/results

# JetEngine parameters
TENSOR_PARALLEL_SIZE=1
MAX_ACTIVE=256
BLOCK_SIZE=4
DENOISING_STEPS=4

# --- Model Definitions ---
BASE_MODEL_DIR="your_path/models"
BASE_MODEL_NAME="your_model_name"

# --- Create list of models to evaluate ---
declare -a models_to_evaluate

# Add base model
models_to_evaluate+=("${BASE_MODEL_DIR}/${BASE_MODEL_NAME}")

# Add additional models as needed
# models_to_evaluate+=("your_path/another_model")

# ======================= EXECUTION =======================
for MODEL_PATH in "${models_to_evaluate[@]}"; do
    echo "=========================================="
    echo "Evaluating model: ${MODEL_PATH}"
    echo "=========================================="
    
    MODEL_NAME=$(basename "${MODEL_PATH}")
    OUTPUT_DIR="${MAIN_OUTPUT_DIR}/${MODEL_NAME}"
    
    python arena_sdar.py \
        --model_name_or_path "${MODEL_PATH}" \
        --dataset_path "${DATA_PATH}" \
        --output_dir "${OUTPUT_DIR}" \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --max_active ${MAX_ACTIVE} \
        --block_size ${BLOCK_SIZE} \
        --denoising_steps ${DENOISING_STEPS}
    
    echo "Finished evaluating: ${MODEL_NAME}"
    echo ""
done

echo "All evaluations complete!"

# ======================= EXECUTION LOOP =======================
echo "Starting evaluation for a series of ${#models_to_evaluate[@]} models..."
echo ""

# Iterate through each model path in the model array
for model_path in "${models_to_evaluate[@]}"; do
    
    # Extract model short name from full path for creating output directory
    model_short_name=$(basename "${model_path}")
    
    # Create an independent output subdirectory for current model
    run_output_dir="${MAIN_OUTPUT_DIR}/${model_short_name}"
    
    # Print log to clearly show which model is currently being evaluated
    echo "========================================================================"
    echo "  STARTING EVALUATION FOR: ${model_short_name}"
    echo "  Model Path: ${model_path}"
    echo "  Output Directory: ${run_output_dir}"
    echo "========================================================================"
    
    # Ensure output directory exists
    mkdir -p "${run_output_dir}"
    
    # Execute evaluation script, passing current model path and independent output directory
    python arena_sdar.py \
        --model_name_or_path "${model_path}" \
        --dataset_path "${DATA_PATH}" \
        --output_dir "${run_output_dir}" \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} \
        --max_active ${MAX_ACTIVE} \
        --block_size ${BLOCK_SIZE} \
        --denoising_steps ${DENOISING_STEPS}

    # Check exit status of previous command
    if [ $? -ne 0 ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "  ERROR: Evaluation failed for model ${model_short_name}. Aborting script."
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        exit 1 # Exit entire script if evaluation fails
    fi
    
    echo "--- Finished evaluation for ${model_short_name} ---"
    echo "" # Add blank line to make log more readable

done

echo "✅ All model evaluations are complete."
