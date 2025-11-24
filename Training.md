基于Instruct模型已有基础能力，我们的训练目标是**让Expert学会分工专业化**

## 🎯 训练方法选择：SFT vs DPO

### 推荐：**先SFT，后DPO（可选）** ⭐⭐⭐⭐⭐

#### 为什么先用SFT？

```
当前状态：
✅ Llama-3.2-3B-Instruct已有instruction能力
✅ Upcycle后，每个Expert都有相同的基础能力

训练目标：
🎯 让8个Expert学会专业化分工
🎯 Router学会任务分配

最佳方法：SFT
```

**SFT的作用**：
```python
# SFT训练过程
for batch in diverse_data:
    # Router根据输入选择2个Expert
    expert1, expert2 = router(input)
    
    # 选中的Expert处理任务并更新
    output = experts[expert1, expert2](input)
    loss = compute_loss(output, label)
    
    # 只有被选中的Expert和Router更新
    # 随着训练，不同Expert会专业化处理不同类型任务
```

#### DPO什么时候用？

**DPO适用于**：
```
场景：模型已经能完成任务，但需要对齐人类偏好
- 选择更有帮助的回复
- 选择更安全的回复
- 选择更符合风格的回复

需要：Preference pairs (好的回复 vs 坏的回复)
```

**对于MoE Upcycling后的模型**：
```
阶段1: SFT (必需)
  → 让Expert分工专业化
  
阶段2: DPO (可选，锦上添花)
  → 进一步对齐偏好
```

---

## 📚 推荐数据集

### 核心原则：**多样性 > 数量**

为了让8个Expert学会专业化，需要**覆盖不同类型任务**的数据。

### 方案A：单一高质量数据集（快速开始）⭐⭐⭐⭐

#### 1. **Alpaca-GPT4** (推荐) ⭐⭐⭐⭐⭐
```yaml
dataset: "vicgalle/alpaca-gpt4"
samples: 52K
quality: 高（GPT-4生成）
diversity: 非常好（instruction, input, output）

优点:
✅ 高质量
✅ 任务多样
✅ 数据干净
✅ 即开即用
```

#### 2. **OpenOrca**
```yaml
dataset: "Open-Orca/OpenOrca"  
samples: 1M+ (可取subset)
quality: 高
diversity: 极好（多种推理任务）

优点:
✅ 包含CoT推理
✅ 任务极其多样
✅ 适合expert专业化
```

### 方案B：混合数据集（最佳效果）⭐⭐⭐⭐⭐

**推荐配置**：混合不同领域，让Expert自然分工

```yaml
# configs/data_mix.yaml
datasets:
  # 1. 通用指令 (30%)
  - name: "vicgalle/alpaca-gpt4"
    weight: 0.3
    samples: 15000
    type: general_instruction
  
  # 2. 对话/助手 (25%)
  - name: "OpenAssistant/oasst2"
    weight: 0.25
    samples: 12000
    type: conversation
  
  # 3. 代码 (20%)
  - name: "iamtarun/python_code_instructions_18k_alpaca"
    weight: 0.2
    samples: 10000
    type: code
  
  # 4. 推理/数学 (15%)
  - name: "gsm8k"
    weight: 0.15
    samples: 7500
    type: reasoning
  
  # 5. 创意写作 (10%)
  - name: "garage-bAInd/Open-Platypus"
    weight: 0.1
    samples: 5000
    type: creative

# Total: ~50K samples
# Training time: ~2-3 hours
```

**为什么这个配置好？**
```
不同类型数据 → 不同Expert专业化

可能的分工（自然涌现）：
Expert 1,2: 通用指令理解
Expert 3,4: 对话和助手任务
Expert 5,6: 代码相关任务
Expert 7,8: 推理和数学任务
```

### 方案C：小规模快速测试 ⭐⭐⭐

```yaml
dataset: "tatsu-lab/alpaca"
samples: 52K
training: ~1 hour

目的: 快速验证pipeline
```

---

## 📊 具体推荐

### 🥇 最推荐：混合数据集

```python
# 创建混合数据配置
{
    "general": {
        "data": "vicgalle/alpaca-gpt4",
        "samples": 15000,
        "description": "通用instruction following"
    },
    "conversation": {
        "data": "OpenAssistant/oasst2", 
        "samples": 12000,
        "description": "多轮对话"
    },
    "code": {
        "data": "iamtarun/python_code_instructions_18k_alpaca",
        "samples": 10000,
        "description": "代码生成"
    },
    "reasoning": {
        "data": "gsm8k",
        "samples": 8000,
        "description": "数学推理"
    }
}

Total: 45K samples
Expected training: 2-3 hours
```

**预期效果**：
- ✅ Expert自然分工处理不同任务
- ✅ Router学会任务分类
- ✅ 模型在各领域都表现良好

---

## 🔄 训练流程建议

### 阶段1：SFT - Expert专业化 (必需)

```bash
# 使用混合数据集SFT训练
python scripts/train_sft.py \
    --model_name_or_path models/Llama-3.2-3B-Instruct-MoE-8x \
    --data_path "混合数据集" \
    --num_experts_to_train 8 \
    --max_steps 5000
```

**时间**：2-3小时  
**效果**：Expert学会分工

### 阶段2：DPO - 偏好对齐 (可选)

```bash
# 如果想进一步优化
python scripts/train_dpo.py \
    --model_name_or_path outputs/sft_checkpoint \
    --data_path "HuggingFaceH4/ultrafeedback_binarized" \
    --max_steps 1000
```

**时间**：1小时  
**效果**：回复更符合人类偏好

---

## 💡 实用建议

### 快速开始方案

**Day 1: 验证pipeline**
```bash
# 1. 下载模型
python scripts/convert_llama3b_to_moe.py

# 2. 小数据集测试
data: "tatsu-lab/alpaca" (前5000样本)
time: 30分钟
goal: 验证代码正常
```

**Day 2: 正式训练**
```bash
# 使用混合数据集
data: 混合配置 (45K samples)
time: 2-3小时
goal: 训练完整模型
```

**Day 3: 评估 & 可选DPO**
```bash
# 评估效果
# 如果满意 → 完成
# 如果需要优化 → DPO
```

---

## 📝 数据集准备示例

让我创建一个数据混合脚本：

```python
# scripts/prepare_mixed_dataset.py
from datasets import load_dataset, concatenate_datasets

def create_mixed_dataset():
    datasets_config = [
        ("vicgalle/alpaca-gpt4", 15000),
        ("OpenAssistant/oasst2", 12000),
        ("iamtarun/python_code_instructions_18k_alpaca", 10000),
        ("gsm8k", 8000),
    ]
    
    mixed = []
    for name, samples in datasets_config:
        ds = load_dataset(name, split=f"train[:{samples}]")
        mixed.append(ds)
    
    final_dataset = concatenate_datasets(mixed)
    final_dataset.shuffle(seed=42)
    final_dataset.save_to_disk("data/mixed_instruction")
    
    return final_dataset
```

---

## 🎯 总结

| 问题 | 答案 |
|------|------|
| **训练方法** | **SFT first** (DPO可选) |
| **最佳数据集** | **混合数据集** (45K samples) |
| **快速开始** | Alpaca-GPT4 (52K) |
| **训练时间** | 2-3小时 |
| **关键目标** | Expert专业化分工 |

**核心策略**：
```
多样化数据 → Expert专业化 → 强大的MoE模型
```

要我帮你准备混合数据集的加载脚本吗？🚀