import sys
import os
import torch
import copy
from tqdm import tqdm
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
        # 使用 low_cpu_mem_usage=True 加速加载并减少内存峰值
        llama_model = AutoModelForCausalLM.from_pretrained(
            llama_path, 
            torch_dtype=torch.bfloat16, 
            device_map="cpu", 
            low_cpu_mem_usage=True
        )
    except OSError:
        print(f"Error: 模型未找到，请先下载 Llama-3.1-8B 到 {llama_path}")
        return

    # 2. 创建 MoE Config
    print("Configuring MoE Model...")
    moe_config = LlamaMoEConfig.from_pretrained(llama_path)
    
    # --- MoE 核心配置 ---
    moe_config.n_routed_experts = 8      # 总 Expert 数量
    moe_config.num_experts_per_tok = 2   # 每次激活 Expert 数量
    
    # 如果显存不够，可以将 moe_intermediate_size 减半
    # moe_config.moe_intermediate_size = 14336 
    moe_config.moe_intermediate_size = 4096 
    
    print(f"Target MoE Intermediate Size: {moe_config.moe_intermediate_size}")
    
    print("Creating MoE Model Structure...")
    # 同样使用 low_cpu_mem_usage=True
    moe_model = LlamaMoEForCausalLM(moe_config)
    moe_model.to(torch.bfloat16)

    # 3. 移植权重 (Sparse Upcycling 核心逻辑)
    print("Transplanting weights (Sparse Upcycling)...")
    
    llama_sd = llama_model.state_dict()
    moe_sd = moe_model.state_dict()
    
    copied_keys = 0
    
    # 使用 tqdm 显示进度
    pbar = tqdm(list(moe_sd.keys()), desc="Copying weights")
    
    with torch.no_grad():
        for key in pbar:
            # 情况 A: Router/Gate 权重 (新增加的层) -> 保持随机初始化
            if "mlp.gate.weight" in key:
                continue
                
            # 情况 B: FFN Expert 权重 -> 从原版 FFN 广播 (Broadcast)
            if "mlp.experts." in key:
                # Key 格式: model.layers.0.mlp.experts.0.gate_proj.weight
                # 目标源:   model.layers.0.mlp.gate_proj.weight
                
                parts = key.split('.')
                layer_idx = parts[2] # "0"
                proj_name = parts[-2] # "gate_proj", "up_proj", "down_proj"
                
                source_key = f"model.layers.{layer_idx}.mlp.{proj_name}.weight"
                
                if source_key in llama_sd:
                    src_tensor = llama_sd[source_key]
                    target_tensor = moe_sd[key]
                    
                    # 检查维度是否需要切片 (Slicing)
                    if src_tensor.shape != target_tensor.shape:
                        if proj_name in ["gate_proj", "up_proj"]:
                            # 切片行
                            target_tensor.data.copy_(src_tensor.data[:target_tensor.shape[0], :])
                        elif proj_name == "down_proj":
                            # 切片列
                            target_tensor.data.copy_(src_tensor.data[:, :target_tensor.shape[1]])
                    else:
                        # 维度一致
                        target_tensor.data.copy_(src_tensor.data)
                    
                    # 添加微小噪声打破对称性
                    noise = torch.randn_like(target_tensor) * 0.001
                    target_tensor.data.add_(noise)
                    
                    copied_keys += 1
                    
            # 情况 C: 其他层 (Attention, Norm, Embed) -> 直接复制
            elif key in llama_sd:
                moe_sd[key].data.copy_(llama_sd[key].data)
                copied_keys += 1

    print(f"Weights transplanted: {copied_keys}/{len(moe_sd)} keys.")

    # 4. 保存
    print(f"Saving Upcycled MoE model to {save_path} ...")
    moe_model.save_pretrained(save_path)
    print("Done! Ready for fine-tuning.")

if __name__ == "__main__":
    upcycle_llama()
