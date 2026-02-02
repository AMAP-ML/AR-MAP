import time
import torch
from transformers import AutoModel, AutoTokenizer

model_path = "Dream-org/Dream-v0-Instruct-7B"
model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to("cuda").eval()

messages = [
    {"role": "user", "content": "Write a story that ends with 'Finally, Joey and Rachel get married.'"}
]
end_ids = tokenizer('Finally, Joey and Rachel get married.', return_tensors="pt").input_ids[0].to(device="cuda")

inputs = tokenizer.apply_chat_template(
    messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
)
input_ids = inputs.input_ids.to(device="cuda")
attention_mask = inputs.attention_mask.to(device="cuda")


# Use a hook function to set the final sentence to be end_ids at each intermediate step
# You can change it to control whatever you want about x in each step with this hook
def generation_tokens_hook_func(step, x, logits):
    x[0, -len(end_ids):] = end_ids
    return x

output = model.diffusion_generate(
    input_ids,
    attention_mask=attention_mask,
    max_new_tokens=256,
    output_history=True,
    return_dict_in_generate=True,
    steps=256,
    temperature=0.6,
    top_p=0.95,
    alg="origin",
    # alg_temp=0.,
    generation_tokens_hook_func=generation_tokens_hook_func
)
generations = [
    tokenizer.decode(g[len(p) :].tolist())
    for p, g in zip(input_ids, output.sequences)
]

print(generations[0].split(tokenizer.eos_token)[0])

# the following lines print the history of generations
history = output.history
for i, h in enumerate(history):
    print(f"############ Step {i} ############")
    time.sleep(0.01)
    print(tokenizer.decode(h[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " "))
    # print(tokenizer.decode(h[0].tolist()))

'''An example generation (maybe different due to randomness)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Write a story that ends with 'Finally, Joey and Rachel get married.'<|im_end|>
<|im_start|>assistant
Once upon a time, there was a young couple named Joey and Rachel. They had a relationship that went back many years. They had been best friends for years, and they had deep feelings for each other.

One day, they went for a walk by the lake. It was a beautiful day, and they found themselves in love other's arms. They knew that they were meant for each other, and they would love each other for the rest of their lives. 

They were a young couple, and they had a lot of plans for a life together. They loved each other deeply, and they wanted to start a life together.

They also had a few challenges to face. They had to deal with their respective families, and they had to balance their careers and relationships. Despite these challenges, they continued to support each other and love and support each other through other.

After many years, Joey and Rachel finally decided to get married. It was an exciting moment for both of them, but they went to the church and got married. Their families and friends were there, and they were happy to be there. They knew that they had started a new chapter of their lives, and they knew that they had each other as their true partner.

Finally, Joey and Rachel get married.
'''