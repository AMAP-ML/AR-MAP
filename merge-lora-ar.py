import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ================= Configuration Paths =================
# 1. Base model path
BASE_MODEL_PATH = "your_path/base_model"

# 2. LoRA weights path
LORA_PATH = "your_path/lora_adapter"

# 3. Output save path
OUTPUT_PATH = "your_path/merged_model"

# 4. Device setting (recommend "cpu" for merging to avoid OOM, merging is fast on CPU)
DEVICE = "cuda:0"
# ===========================================

def main():
    print(f"--- LoRA Model Merging Tool ---")
    print(f"Base: {BASE_MODEL_PATH}")
    print(f"LoRA: {LORA_PATH}")
    print(f"Save: {OUTPUT_PATH}")
    print("-" * 30)

    # 1. Load Base model
    print(f"\n1. Loading Base model...")
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            torch_dtype=torch.bfloat16,  # Keep bf16 precision
            device_map=DEVICE,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"❌ Base model loading failed: {e}")
        return

    # 2. Load Tokenizer
    # Usually load from Base path, if LoRA training modified vocab, Peft will handle it automatically
    print(f"2. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_PATH,
        trust_remote_code=True
    )

    # 3. Load and apply LoRA
    print(f"\n3. Loading LoRA Adapter and mounting...")
    try:
        # Wrap Base model with PeftModel
        model = PeftModel.from_pretrained(
            base_model,
            LORA_PATH,
            torch_dtype=torch.bfloat16,
            device_map=DEVICE
        )
        print("   LoRA mounted successfully!")
    except Exception as e:
        print(f"❌ LoRA loading failed. Please check if adapter_config.json exists. Error: {e}")
        return

    # 4. Execute merge (Merge and Unload)
    print(f"\n4. Starting weight merging (W_new = W_base + W_lora * scaling)...")
    # merge_and_unload will add LoRA weights directly to Base weights and remove LoRA modules
    model = model.merge_and_unload()
    print("   Merging complete! Model has been converted back to standard architecture.")

    # 5. Save complete model
    print(f"\n5. Saving final model to: {OUTPUT_PATH}")
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Save weights (automatically handle as safetensors or bin)
    model.save_pretrained(OUTPUT_PATH, safe_serialization=True)
    
    # Save Tokenizer
    tokenizer.save_pretrained(OUTPUT_PATH)

    print(f"\n✅ All done!")
    print(f"Output path: {OUTPUT_PATH}")
    print("You can directly load this directory for inference.")

if __name__ == "__main__":
    main()