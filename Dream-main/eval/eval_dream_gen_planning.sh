
model=Dream-org/Dream-v0-Base-7B

CUDA_VISIBLE_DEVICES=6 python eval_planning.py task=sudoku prediction_path=eval_results/sudoku.jsonl ckpt=$model \
    gen_params.diffusion_steps=24 \
    gen_params.max_new_tokens=24 \
    gen_params.temperature=0 \
    gen_params.top_p=1 \
    gen_params.alg=entropy \
    gen_params.alg_temp=0

CUDA_VISIBLE_DEVICES=6 python eval_planning.py task=cd4 prediction_path=eval_results/cd4.jsonl ckpt=$model \
    gen_params.diffusion_steps=32 \
    gen_params.max_new_tokens=32 \
    gen_params.temperature=0 \
    gen_params.top_p=1  \
    gen_params.alg=entropy \
    gen_params.alg_temp=0

CUDA_VISIBLE_DEVICES=6 python eval_planning.py task=trip prediction_path=eval_results/trip.jsonl ckpt=$model \
    gen_params.diffusion_steps=256 \
    gen_params.max_new_tokens=256 \
    gen_params.temperature=0 \
    gen_params.top_p=1  \
    gen_params.alg=entropy \
    gen_params.alg_temp=0
