import sys
import os
import torch
import copy
from transformers import AutoModelForCausalLM, AutoConfig

# 使得脚本可以在 scripts/ 目录下直接运行，也能找到 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.configuration_llama_moe import LlamaMoEConfig
from src.modeling_llama_moe import LlamaMoEForCausalLM

def upcycle_llama():
    # 1. 配置路径
    llama_path = "models/Llama-3.1-8B"
    save_path = "models/Llama-3.1-8B-MoE-Upcycled"
    
    print(f"Loading Llama-3 from {llama_path}...")
    try:
        llama_model = AutoModelForCausalLM.from_pretrained(llama_path, torch_dtype=torch.bfloat16, device_map="cpu")
    except OSError:
        print(f"Error: 模型未找到，请先下载 Llama-3.1-8B 到 {llama_path}")
        return

    # 2. 创建 MoE Config
    print("Configuring MoE Model...")
    moe_config = LlamaMoEConfig.from_pretrained(llama_path)
    
    # --- MoE 核心配置 ---
    moe_config.n_routed_experts = 8      # 总 Expert 数量
    moe_config.num_experts_per_tok = 2   # 每次激活 Expert 数量
    
    # 注意：Llama-3-8B 原版 intermediate_size 为 14336
    # 如果保持 14336 * 8 个 Experts，模型总参数量将超过 40B，需要巨大的显存。
    # 如果想节省显存，可以将其设为 4096 或更小 (会自动进行权重切片)
    # moe_config.moe_intermediate_size = 14336 
    moe_config.moe_intermediate_size = 4096 # 如果显存不够，取消注释这行使用切片初始化
    
    print(f"Target MoE Intermediate Size: {moe_config.moe_intermediate_size}")
    
    print("Creating MoE Model Structure...")
    moe_model = LlamaMoEForCausalLM(moe_config)
    moe_model.to(torch.bfloat16)

    # 3. 移植权重 (Sparse Upcycling 核心逻辑)
    print("Transplanting weights (Sparse Upcycling)...")
    
    llama_sd = llama_model.state_dict()
    moe_sd = moe_model.state_dict()
    
    # 统计信息
    total_keys = len(moe_sd)
    copied_keys = 0
    
    for key in moe_sd.keys():
        # 情况 A: Router/Gate 权重 (新增加的层) -> 保持随机初始化
        if "mlp.gate.weight" in key:
            # print(f"  [Init] Randomly initializing router: {key}")
            continue
            
        # 情况 B: FFN Expert 权重 -> 从原版 FFN 广播 (Broadcast)
        if "mlp.experts." in key:
            # Key 格式: model.layers.0.mlp.experts.0.gate_proj.weight
            # 目标源:   model.layers.0.mlp.gate_proj.weight
            
            # 解析层号和投影类型
            parts = key.split('.')
            layer_idx = parts[2] # "0"
            proj_name = parts[-2] # "gate_proj", "up_proj", "down_proj"
            
            source_key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
            
            if source_key in llama_sd:
                src_tensor = llama_sd[source_key]
                target_tensor = moe_sd[key]
                
                # 检查维度是否需要切片 (Slicing)
                if src_tensor.shape != target_tensor.shape:
                    # print(f"  [Slice] {key}: {src_tensor.shape} -> {target_tensor.shape}")
                    # Linear 权重形状为 [out_features, in_features]
                    if proj_name in ["gate_proj", "up_proj"]:
                        # 输出维度变化 (intermediate_size)，输入维度不变 (hidden_size) -> 切片行
                        target_tensor.data.copy_(src_tensor.data[:target_tensor.shape[0], :])
                    elif proj_name == "down_proj":
                        # 输入维度变化 (intermediate_size)，输出维度不变 (hidden_size) -> 切片列
                        target_tensor.data.copy_(src_tensor.data[:, :target_tensor.shape[1]])
                else:
                    # 维度一致，直接复制
                    target_tensor.data.copy_(src_tensor.data)
                
                # 添加微小噪声打破对称性 (让不同 Expert 在训练初期就有细微差别)
                noise = torch.randn_like(target_tensor) * 0.001
                target_tensor.data.add_(noise)
                
                copied_keys += 1
            else:
                print(f"Warning: Source key {source_key} not found for expert layer")
                
        # 情况 C: 其他层 (Attention, Norm, Embed) -> 直接复制
        elif key in llama_sd:
            moe_sd[key].data.copy_(llama_sd[key].data)
            copied_keys += 1
        else:
            print(f"Warning: Key {key} not found in Llama-3 model")

    print(f"Weights transplanted: {copied_keys}/{total_keys} keys. (Gate weights remain random)")

    # 4. 保存
    print(f"Saving Upcycled MoE model to {save_path} ...")
    moe_model.save_pretrained(save_path)
    print("Done! Ready for fine-tuning.")

if __name__ == "__main__":
    upcycle_llama()
