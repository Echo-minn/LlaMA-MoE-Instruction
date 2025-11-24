#!/usr/bin/env python3
"""
估算不同MoE配置的显存占用
"""

def estimate_memory(base_params_b, num_experts, expert_size_b, 
                    use_qlora=True, batch_size=4, seq_length=1024):
    """
    估算训练显存占用
    
    参数:
        base_params_b: Base模型参数量(B)
        num_experts: Expert数量
        expert_size_b: 每个Expert参数量(B)
        use_qlora: 是否使用QLoRA
        batch_size: 每GPU的batch size
        seq_length: 序列长度
    """
    print(f"\n{'='*70}")
    print(f"配置: {base_params_b}B Base + {num_experts} Experts × {expert_size_b}B")
    print(f"训练: {'QLoRA' if use_qlora else 'Full LoRA'}, batch={batch_size}, seq_len={seq_length}")
    print(f"{'='*70}")
    
    # 总参数量
    total_params_b = base_params_b + num_experts * expert_size_b
    print(f"\n📊 参数量:")
    print(f"  Base模型:     {base_params_b:.1f}B")
    print(f"  {num_experts} Experts:   {num_experts * expert_size_b:.1f}B ({expert_size_b:.2f}B each)")
    print(f"  总计:         {total_params_b:.1f}B")
    
    # 激活参数（每次2个expert）
    active_params_b = base_params_b + 2 * expert_size_b
    print(f"  激活参数:     {active_params_b:.1f}B (每次推理)")
    
    # 显存计算
    print(f"\n💾 显存占用估算 (单GPU):")
    
    # 1. 模型参数
    if use_qlora:
        model_memory_gb = total_params_b * 0.5  # 4-bit = 0.5 bytes per param
        param_note = "4-bit量化"
    else:
        model_memory_gb = total_params_b * 2  # bf16 = 2 bytes per param
        param_note = "bf16"
    print(f"  模型参数 ({param_note}):     {model_memory_gb:.2f} GB")
    
    # 2. LoRA适配器
    lora_params_b = total_params_b * 0.02  # 假设LoRA是2%参数量
    lora_memory_gb = lora_params_b * 2  # bf16
    print(f"  LoRA适配器 (bf16):      {lora_memory_gb:.2f} GB")
    
    # 3. Optimizer状态（只优化LoRA）
    optimizer_memory_gb = lora_memory_gb * 2  # AdamW需要2倍参数内存
    print(f"  Optimizer状态:         {optimizer_memory_gb:.2f} GB")
    
    # 4. 梯度
    gradient_memory_gb = lora_memory_gb
    print(f"  梯度缓存:              {gradient_memory_gb:.2f} GB")
    
    # 5. 激活值（与batch size和seq length相关）
    # 粗略估算: 每个token每B参数约需要 4 bytes (bf16 中间激活)
    activation_memory_gb = active_params_b * batch_size * seq_length * 4 / 1e9
    print(f"  激活值 (batch={batch_size}):     {activation_memory_gb:.2f} GB")
    
    # 6. KV cache
    hidden_size = int(base_params_b * 1000)  # 粗略估算
    kv_memory_gb = 2 * batch_size * seq_length * hidden_size * 2 / 1e9
    print(f"  KV cache:              {kv_memory_gb:.2f} GB")
    
    # 7. 其他开销
    other_memory_gb = 1.0
    print(f"  其他开销:              {other_memory_gb:.2f} GB")
    
    # 总计
    total_memory_gb = (model_memory_gb + lora_memory_gb + optimizer_memory_gb + 
                       gradient_memory_gb + activation_memory_gb + kv_memory_gb + 
                       other_memory_gb)
    
    print(f"\n  {'─'*66}")
    print(f"  总计 (单GPU峰值):      {total_memory_gb:.2f} GB")
    
    # ZeRO-2分布式
    print(f"\n🔧 ZeRO-2 分布式 (4 GPUs):")
    # ZeRO-2分片optimizer和gradient
    per_gpu_memory_gb = (model_memory_gb + lora_memory_gb + 
                         optimizer_memory_gb/4 + gradient_memory_gb/4 +
                         activation_memory_gb + kv_memory_gb + other_memory_gb)
    print(f"  每卡显存:              {per_gpu_memory_gb:.2f} GB")
    print(f"  4卡总显存:             {per_gpu_memory_gb * 4:.2f} GB")
    
    # 判断可行性
    print(f"\n✅ 可行性分析 (每卡40GB):")
    if per_gpu_memory_gb < 30:
        status = "✅ 绰绰有余"
        detail = f"还剩 {40 - per_gpu_memory_gb:.1f}GB，可增大batch size"
    elif per_gpu_memory_gb < 38:
        status = "✅ 完全可行"
        detail = f"还剩 {40 - per_gpu_memory_gb:.1f}GB"
    elif per_gpu_memory_gb < 40:
        status = "🟡 可行但紧张"
        detail = "建议减小batch size或启用更多优化"
    else:
        status = "❌ 可能OOM"
        detail = "需要减小batch size或使用更多优化"
    
    print(f"  {status}")
    print(f"  {detail}")
    
    # 速度估算
    print(f"\n⚡ 训练速度估算:")
    # 简单模型: 时间 ∝ (total_params)^1.3 × seq_length^1.5 / batch_size
    base_time = (total_params_b ** 1.3) * (seq_length / 1000) ** 1.5 / batch_size * 0.5
    print(f"  每iteration:           ~{base_time:.1f}s")
    print(f"  2000 steps:            ~{base_time * 2000 / 3600:.1f} 小时")
    
    return per_gpu_memory_gb

print("="*70)
print("Llama-3B-MoE 显存占用估算")
print("="*70)

# 配置1: 3B + 4 experts
mem1 = estimate_memory(
    base_params_b=3.0,
    num_experts=4,
    expert_size_b=0.75,
    use_qlora=True,
    batch_size=4,
    seq_length=1024
)

# 配置2: 3B + 8 experts
mem2 = estimate_memory(
    base_params_b=3.0,
    num_experts=8,
    expert_size_b=0.75,
    use_qlora=True,
    batch_size=4,
    seq_length=1024
)

# 对比
print(f"\n{'='*70}")
print("📊 对比总结")
print(f"{'='*70}")
print(f"\n配置对比:")
print(f"  {'配置':<20} {'每卡显存':<15} {'训练速度':<15} {'推荐度'}")
print(f"  {'-'*66}")
print(f"  {'3B + 4 experts':<20} {mem1:.1f} GB{' '*7} {'更快':<15} ⭐⭐⭐⭐⭐")
print(f"  {'3B + 8 experts':<20} {mem2:.1f} GB{' '*7} {'稍慢':<15} ⭐⭐⭐⭐")

print(f"\n💡 建议:")
print(f"  • 目标是Instruction Following → 选择 3B+4experts")
print(f"  • 追求最强多任务能力 → 选择 3B+8experts")
print(f"  • 两者都可在4卡40GB上训练 ✅")
print(f"\n{'='*70}")

