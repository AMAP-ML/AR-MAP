import torch
from transformers import AutoModel, AutoTokenizer


def get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available() and mps_backend.is_built():
        return "mps"
    return "cpu"


def get_preferred_dtype(device):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32
import time


def _sanitize_probabilities(probs: torch.Tensor) -> torch.Tensor:
    """Keep multinomial inputs finite, non-negative, and normalized."""
    if not torch.is_floating_point(probs):
        return probs
    sanitized = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    sanitized = sanitized.clamp_min(0.0)
    if sanitized.ndim == 1:
        total = sanitized.sum()
        if total <= 0:
            return torch.full_like(sanitized, 1.0 / sanitized.numel())
        return sanitized / total
    total = sanitized.sum(dim=-1, keepdim=True)
    zero_mask = total <= 0
    if zero_mask.any():
        sanitized = sanitized.clone()
        sanitized[zero_mask.squeeze(-1)] = 1.0
        total = sanitized.sum(dim=-1, keepdim=True)
    return sanitized / total


_original_multinomial = torch.multinomial


def _safe_multinomial(probs, num_samples, replacement=False, *, generator=None, out=None):
    normalized = _sanitize_probabilities(probs)
    return _original_multinomial(normalized, num_samples, replacement, generator=generator, out=out)


if not getattr(torch.multinomial, "_dream_safe", False):
    _safe_multinomial._dream_safe = True  # type: ignore[attr-defined]
    torch.multinomial = _safe_multinomial  # type: ignore[assignment]
# Load model and tokenizer
model_path = "Dream-org/Dream-v0-Instruct-7B"
device = get_best_device()
dtype = get_preferred_dtype(device)
model = AutoModel.from_pretrained(model_path, torch_dtype=dtype, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = model.to(device).eval()

# Initialize conversation history
messages = []

print("Multi-turn conversation with Dream-v0-Instruct-7B")
print("Type 'exit' to end the conversation")
print("----------------------------------------------")

while True:
    # Get user input
    user_input = input("You: ")

    # Check if user wants to exit
    if user_input.lower() == 'exit':
        print("Conversation ended.")
        break

    # Add user message to conversation history
    messages.append({"role": "user", "content": user_input})

    # Format input with chat template
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
    )
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    def generation_tokens_hook_func(step, x, logits):
        print(f"############ Step {step} ############")
        # print(tokenizer.decode(h[0].tolist()))
        print(tokenizer.decode(x[0].tolist()).split(tokenizer.eos_token)[0].replace(tokenizer.mask_token, " "), end="\r")
        time.sleep(0.01)
        return x
    
    # Generate response
    output = model.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=256,
        output_history=True,
        return_dict_in_generate=True,
        steps=256,
        temperature=0.4,
        top_p=0.95,
        alg="entropy",
        alg_temp=0.1,
        generation_tokens_hook_func=generation_tokens_hook_func
    )

    # Process response
    generation = tokenizer.decode(output.sequences[0][len(input_ids[0]):].tolist())
    generation = generation.split(tokenizer.eos_token)[0].strip()

    # Print response
    print("Model:", generation)

    # Add model response to conversation history
    messages.append({"role": "assistant", "content": generation})


'''An example conversation (maybe different due to randomness)
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?<|im_end|>
<|im_start|>assistant
Janet sells 16 - 3 - 4 = 9 eggs per day.
She makes 9 * $2 = $18 per day.<|im_end|>
<|im_start|>user
what if her duck lay three more eggs<|im_end|>
<|im_start|>assistant
If Janet's ducks lay three more eggs per day, she would have 16 + 3 = 19 eggs per day.<|im_end|>
<|im_start|>user
yes, so how many dollars she make<|im_end|>
<|im_start|>assistant
Janet sells 19 - 3 - 4 = 12 eggs per day.
She makes 12 * $2 = $24 per day.
'''