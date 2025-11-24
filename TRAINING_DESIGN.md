Based on the existing capabilities of the Instruct model(Llama-3.2-3B-Instruct), our training goal is to **let Expert specialize in different tasks**

## Training Method Selection: SFT vs DPO

### Recommended: **SFT first, then DPO (optional)**

#### Training Goal:

```
 Let 8 Experts specialize in different tasks
 Router learns task allocation
```

**Core Strategy**:
```
Diverse data → Expert specialization → Powerful MoE model
Base model: Llama-3.2-3B-Instruct
MoE model: Llama-3.2-3B-Instruct-MoE-8x
Method: QLoRA (4-bit)
LoRA rank: 64
Training Experts: All 8, Actived 2 per layer
Batch size: 32 (4 GPUs × 8 × 1)
Learning rate: 2e-5
Sequence length: 1024 (max_length)
```

## Experiment Design

### Step 1: SFT - Expert specialization (Required)

### Step 2: In-context Learning - Expert specialization (Optional)

### Step 3: DPO, for Preference alignment (Optional)

## Quick Start Pipeline

**验证pipeline**
```bash
# 1. Download the model and convert to MoE format(broadcast FFN to 8 experts)
python scripts/convert_llama3b_to_moe.py

# 2. validation
data: "tatsu-lab/alpaca" (5000 samples)
time: 30 minutes
goal: Verify the code and training pipeline is working
```

**QLoRA + SFT**
```bash

data: Mixed Dataset(45K samples)
time: 11 hours
goal: Train the full model
```

## In-context learning

Llama-3.2-3B VS Llama-3.2-3B-Instruction
System prompts for different task's high quality outputs
Prevent ethical, harmful, biased output
Evaluate on same evaluation dataset and benchmark

**DPO(optional)**
```bash
# TODO for better huaman preference alignment
```

**Evaluation Design**

GSM8K：Accuracy => NO
HumanEval：Pass@1, pass@10
MMLU

one or two task evaluation

## Distributed training

DP(ZeRO2) + TP

## Parameter and Dataset Configuration

- Expert specialize in different tasks naturally
- Router learns task classification
- The model performs well in all domains

### Model Parameters Size and learnable parameters

trainable params: 344,981,504 || all params: 9,900,207,104 || trainable%: 3.4846
8 experts, activated 2 experts per layer

### Dataset Composition
================================================================================
Loading vicgalle/alpaca-gpt4 (alpaca format, 15000 samples)...
 ✅ Loaded 15000 valid samples from vicgalle/alpaca-gpt4
Loading OpenAssistant/oasst2 (oasst format, 12000 samples)...
 ✅ Loaded 0 valid samples from OpenAssistant/oasst2
Loading iamtarun/python_code_instructions_18k_alpaca (alpaca format, 10000 samples)...
 ✅ Loaded 10000 valid samples from iamtarun/python_code_instructions_18k_alpaca
Loading gsm8k (gsm8k format, 8000 samples)...
 ✅ Loaded 7473 valid samples from gsm8k
Loading garage-bAInd/Open-Platypus (alpaca format, 5000 samples)...
 ✅ Loaded 5000 valid samples from garage-bAInd/Open-Platypus
 📊 Mixing 5 dataset(s)...
 ✅ Final mixed dataset: 37473 samples
================================================================================
Train samples: 35599
Eval samples: 1874

### Mixed Dataset Configuration

**Mix different domains, let Expert specialize naturally**

```yaml
# configs/data_mix_instruction.yaml
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

```

**Why this configuration?**
```
Different types of data → Different Expert specialization

Possible specialization (emergent):
Expert 1,2: General instruction understanding
Expert 3,4: Conversation and assistant tasks
Expert 5,6: Code-related tasks
Expert 7,8: Reasoning and mathematics tasks
```

## Summary

**Core Strategy**:
```
Diverse data → Expert specialization → Powerful MoE model
Base model: Llama-3.2-3B-Instruct
Output model: Llama-3.2-3B-Instruct-MoE-8x
Method: QLoRA (4-bit)
LoRA rank: 64
Training Experts: All 8, Actived 2 per layer
Batch size: 32 (4 GPUs × 8 × 1)
Learning rate: 2e-5
Sequence length: 1024 (max_length)
```
