
tasks="mmlu arc_easy arc_challenge hellaswag piqa gpqa_main_n_shot winogrande race"
nshots="5 0 0 0 0 5 5 0"
# tasks="mmlu"
# nshots="5"

# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"

# Iterate through the arrays
for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_results/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    accelerate launch --main_process_port 29510 eval.py --model dream \
        --model_args pretrained=Dream-org/Dream-v0-Base-7B,add_bos_token=true \
        --tasks ${TASKS_ARRAY[$i]} \
        --batch_size 32 \
        --output_path $output_path \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --log_samples \
        --confirm_run_unsafe_code 
done
