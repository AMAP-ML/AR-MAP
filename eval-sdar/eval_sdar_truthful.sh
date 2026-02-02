#!/bin/bash

# Script will exit immediately if any command fails
set -e

# --- Configuration ---

# [!] Fill in your model paths to evaluate here [!]
# Each model path on a separate line, enclosed in quotes
MODELS_TO_EVALUATE=(
    "your_path/model1"
    "your_path/model2"
    # Add more model paths here...
)

# Base output directory, all evaluation results will be saved in subfolders
BASE_OUTPUT_DIR="all_results"

# Number of GPUs
NUM_GPUS=8

# SDAR Block Size (must match training configuration)
BLOCK_SIZE=1

# Port (to prevent multi-task conflicts)
MASTER_PORT=29505

# ----------------------------------------

echo "Starting batch evaluation for ${#MODELS_TO_EVALUATE[@]} models..."

# Ensure base output directory exists
mkdir -p "$BASE_OUTPUT_DIR"

# Loop through model list
for MODEL_PATH in "${MODELS_TO_EVALUATE[@]}"; do
    
    # Check if model path exists
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model path not found, skipping: $MODEL_PATH"
        continue
    fi

    # Extract model name from model path to create unique output directory
    # Example: /path/to/MyModel-v1 -> MyModel-v1
    MODEL_NAME=$(basename "$MODEL_PATH")
    
    # Set unique output directory for current model
    OUTPUT_DIR="$BASE_OUTPUT_DIR/${MODEL_NAME}_truthful_mc2"

    echo "======================================================================"
    echo "=> Evaluating Model: $MODEL_NAME"
    echo "=> Model Path:       $MODEL_PATH"
    echo "=> Outputting to:    $OUTPUT_DIR"
    echo "======================================================================"

    # Launch evaluation script using accelerate
    # --mixed_precision bf16: Enable half-precision acceleration
    accelerate launch \
        --num_processes $NUM_GPUS \
        --main_process_port $MASTER_PORT \
        --mixed_precision bf16 \
        sdar_truthful.py \
        --model_name_or_path "$MODEL_PATH" \
        --output_dir "$OUTPUT_DIR" \
        --block_size "$BLOCK_SIZE" \
        --max_examples 10000 \
        --dtype bf16
    
    echo "✅ Finished evaluation for $MODEL_NAME."
    echo "Results saved to $OUTPUT_DIR"
    echo "" # Add blank line to improve readability

done

echo "🎉 All evaluations finished!"
echo "All results are located in subdirectories under '$BASE_OUTPUT_DIR'."
