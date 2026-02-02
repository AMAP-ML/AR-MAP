import os
import json
import shutil
import torch
import safetensors.torch
from tqdm import tqdm
import argparse

def merge_lora_to_sharded_model(base_model_path, lora_adapter_path, output_path, merging_weight=1.0):
    """
    将 LoRA 权重合并到 sharded (分片) 的基础模型中。

    Args:
        base_model_path (str): 基础模型的路径，包含 sharded safetensors 和 index.json。
        lora_adapter_path (str): LoRA 适配器路径，包含 adapter_model.safetensors 和 adapter_config.json。
        output_path (str): 合并后模型的保存路径。
        merging_weight (float): LoRA 权重的合并系数 (你提到的“倍速”)。
    """
    print(f"开始合并模型...")
    print(f"  - 基础模型: {base_model_path}")
    print(f"  - LoRA 适配器: {lora_adapter_path}")
    print(f"  - 输出目录: {output_path}")
    print(f"  - 合并权重系数: {merging_weight}")

    # --- 1. 创建输出目录并复制非权重文件 ---
    print("\n[步骤 1/6] 创建输出目录并复制配置文件...")
    os.makedirs(output_path, exist_ok=True)
    
    files_to_copy = [f for f in os.listdir(base_model_path) if not f.endswith(('.safetensors', '.bin', '.json'))]
    # 单独处理json文件，避免覆盖我们即将生成的新index.json
    json_files = [f for f in os.listdir(base_model_path) if f.endswith('.json') and 'safetensors' not in f]
    files_to_copy.extend(json_files)

    for file_name in files_to_copy:
        source_file = os.path.join(base_model_path, file_name)
        destination_file = os.path.join(output_path, file_name)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)
    print(f"已复制 {len(files_to_copy)} 个配置文件。")


    # --- 2. 加载 LoRA 权重和配置 ---
    print("\n[步骤 2/6] 加载 LoRA 权重和配置...")
    try:
        lora_config_path = os.path.join(lora_adapter_path, "adapter_config.json")
        with open(lora_config_path, "r") as f:
            lora_config = json.load(f)
        
        lora_alpha = lora_config["lora_alpha"]
        lora_r = lora_config["r"]
        scaling = (lora_alpha / lora_r) * merging_weight
        print(f"  - LoRA alpha: {lora_alpha}, r: {lora_r}")
        print(f"  - 最终缩放因子 (alpha/r * merging_weight): {scaling}")

        lora_weights_path = os.path.join(lora_adapter_path, "adapter_model.safetensors")
        lora_weights = safetensors.torch.load_file(lora_weights_path, device="cuda:0")
    except FileNotFoundError as e:
        print(f"错误: 找不到 LoRA 文件: {e.filename}。请检查路径。")
        return
        
    # --- 3. 组织 LoRA 权重以便查找 ---
    print("\n[步骤 3/6] 组织 LoRA 权重...")
    lora_layer_map = {}
    for key, tensor in lora_weights.items():
        if "lora_A" in key:
            # e.g., "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
            # -> "layers.0.self_attn.q_proj"
            
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # !!! 这是关键的修改点 !!!
            # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # 将 'base_model.model.' 替换为空字符串 ''，以匹配没有此前缀的基础模型。
            target_layer_name = key.split('.lora_A.')[0].replace('base_model.model.', '')
            
            lora_b_key = key.replace("lora_A", "lora_B")
            if lora_b_key in lora_weights:
                lora_layer_map[target_layer_name] = {
                    "lora_A": tensor,
                    "lora_B": lora_weights[lora_b_key]
                }
    print(f"找到了 {len(lora_layer_map)} 个 LoRA 层需要合并。")
    
    # --- 4. Load base model shard index ---
    print("\n[Step 4/6] Loading base model weight index...")
    base_model_index_path = os.path.join(base_model_path, "model.safetensors.index.json")
    try:
        with open(base_model_index_path, "r") as f:
            base_model_index = json.load(f)
    except FileNotFoundError:
        print(f"Error: Cannot find base model index file: {base_model_index_path}.")
        print("Please confirm the base model is in sharded safetensors format.")
        return
        
    weight_map = base_model_index["weight_map"]
    
    # Group weights by shard file
    shards = {}
    for weight_name, shard_file in weight_map.items():
        if shard_file not in shards:
            shards[shard_file] = []
        shards[shard_file].append(weight_name)
        
    # --- 5. Iterate through shards, merge weights and save ---
    print(f"\n[Step 5/6] Iterating through {len(shards)} shards and merging weights...")
    processed_lora_layers = set()
    for shard_file, weights_in_shard in tqdm(shards.items(), desc="Merging shards"):
        shard_path = os.path.join(base_model_path, shard_file)
        # Load current shard weights
        shard_weights = safetensors.torch.load_file(shard_path, device="cuda:0")
        
        # Iterate through all weights in this shard
        for weight_name in weights_in_shard:
            # Weight names are usually '...proj.weight', while LoRA map key is '...proj'
            if weight_name.endswith(".weight"):
                # weight_name could be 'model.layers.0...', or 'layers.0...'
                # We need to find the matching part in lora_layer_map
                
                # Try two matching methods: with 'model.' prefix and without
                possible_targets = [
                    weight_name.replace(".weight", ""), # e.g. "model.layers.0.self_attn.q_proj"
                    weight_name.replace(".weight", "").replace("model.", "", 1) # e.g. "layers.0.self_attn.q_proj"
                ]

                target_layer_name = None
                for t in possible_targets:
                    if t in lora_layer_map:
                        target_layer_name = t
                        break

                if target_layer_name:
                    # Found layer to merge
                    lora_A = lora_layer_map[target_layer_name]["lora_A"]
                    lora_B = lora_layer_map[target_layer_name]["lora_B"]
                    base_weight = shard_weights[weight_name]
                    
                    # Calculate LoRA delta
                    # Use .float() for calculation to ensure precision, then convert back to original type
                    delta = (lora_B.float() @ lora_A.float()) * scaling
                    
                    # Merge weights
                    shard_weights[weight_name] = base_weight + delta.to(base_weight.dtype)
                    
                    processed_lora_layers.add(target_layer_name)

        # Save merged new shard
        output_shard_path = os.path.join(output_path, shard_file)
        safetensors.torch.save_file(shard_weights, output_shard_path)
        
        # Free GPU memory
        del shard_weights
        torch.cuda.empty_cache()

    print(f"Processed {len(processed_lora_layers)} LoRA layers.")
    if len(processed_lora_layers) != len(lora_layer_map):
        unprocessed_count = len(lora_layer_map) - len(processed_lora_layers)
        print(f"Warning: {unprocessed_count} LoRA layers did not find corresponding weights in base model, please check if models match.")
    
    # --- 6. Create new index.json ---
    print("\n[Step 6/6] Creating new model index file...")
    new_index = base_model_index.copy()
    # weight_map remains unchanged as it only maps weight names to file names, and we kept the file names
    new_index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(new_index_path, "w") as f:
        json.dump(new_index, f, indent=4)
        
    print("\n🎉 Merging complete!")
    print(f"Merged model saved to: {output_path}")

if __name__ == '__main__':
    # --- Configure your paths ---
    
    # Base model path (directory containing model.safetensors.index.json)
    BASE_MODEL_PATH = "your_path/base_model" 
    
    # LoRA weights path
    LORA_ADAPTER_PATH = "your_path/lora_adapter"
    
    # Output path (a new directory)
    OUTPUT_PATH = "your_path/merged_model"
    
    # Merging weight coefficient
    MERGING_WEIGHT = 6.0

    # Use argparse to allow modifying paths from command line
    parser = argparse.ArgumentParser(description="Merge LoRA weights into sharded base model.")
    parser.add_argument("--base_model", type=str, default=BASE_MODEL_PATH, help="Path to base model.")
    parser.add_argument("--lora_adapter", type=str, default=LORA_ADAPTER_PATH, help="Path to LoRA adapter.")
    parser.add_argument("--output", type=str, default=OUTPUT_PATH, help="Output path for merged model.")
    parser.add_argument("--weight", type=float, default=MERGING_WEIGHT, help="LoRA merging weight coefficient.")

    args = parser.parse_args()

    # Check if base model path is correct
    if not os.path.exists(os.path.join(args.base_model, "model.safetensors.index.json")):
        print(f"Error: Cannot find 'model.safetensors.index.json' in '{args.base_model}'.")
        print("Please modify 'BASE_MODEL_PATH' in the script to your model's correct path.")
    else:
        merge_lora_to_sharded_model(
            base_model_path=args.base_model,
            lora_adapter_path=args.lora_adapter,
            output_path=args.output,
            merging_weight=args.weight
        )