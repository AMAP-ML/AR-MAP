#!/bin/bash


export PYTHONPATH="${PYTHONPATH}:/mnt/workspace/common/models" 


MODEL_PATH="your_path/base_model"


OUTPUT_DIR="your_dir_here"


export CUDA_VISIBLE_DEVICES=7

echo "Starting TruthfulQA MC2 Evaluation for Dream Model..."
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"


python dream-truthful.py \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --batch_size 8

echo "Evaluation Finished."