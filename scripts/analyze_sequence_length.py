#!/usr/bin/env python3
"""
快速分析数据集的序列长度分布
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer
from datasets import load_dataset
import numpy as np

# 配置
MODEL_PATH = "models/Llama-3.1-8B-MoE-Upcycled"
DATA_PATH = "facebook/natural_reasoning"
SAMPLE_SIZE = 1000  # 分析前1000个样本

print("=" * 80)
print("序列长度分析")
print("=" * 80)

# 加载tokenizer
print(f"\n📥 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    padding_side="right",
    use_fast=True,
    trust_remote_code=True
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✅ Tokenizer加载成功")

# 加载数据集
print(f"\n📥 加载数据集（前{SAMPLE_SIZE}个样本）...")
dataset = load_dataset(DATA_PATH, split=f"train[:{SAMPLE_SIZE}]")
print(f"✅ 数据集加载成功: {len(dataset)} 个样本")

# 分析长度
print(f"\n📊 分析序列长度...")
instruction_lengths = []
response_lengths = []
total_lengths = []

for i, sample in enumerate(dataset):
    if i % 100 == 0:
        print(f"  处理进度: {i}/{len(dataset)}", end="\r")
    
    # 提取question和response
    question = sample.get("question", "")
    
    response = ""
    if "responses" in sample:
        resp_data = sample["responses"]
        if isinstance(resp_data, list) and len(resp_data) > 0:
            first_response = resp_data[0]
            if isinstance(first_response, dict) and "response" in first_response:
                response = first_response["response"]
            else:
                response = str(first_response)
    elif "reference_answer" in sample:
        response = sample["reference_answer"]
    
    # Tokenize
    instruction_text = f"### Instruction:\n{question}\n\n### Response:\n"
    instruction_tokens = tokenizer(instruction_text, add_special_tokens=True)["input_ids"]
    response_tokens = tokenizer(response + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    
    instruction_lengths.append(len(instruction_tokens))
    response_lengths.append(len(response_tokens))
    total_lengths.append(len(instruction_tokens) + len(response_tokens))

print(f"\n  处理进度: {len(dataset)}/{len(dataset)} ✅")

# 统计分析
def print_stats(name, lengths):
    arr = np.array(lengths)
    print(f"\n{'='*60}")
    print(f"{name}")
    print(f"{'='*60}")
    print(f"  样本数量: {len(arr)}")
    print(f"  平均值:   {arr.mean():.1f} tokens")
    print(f"  中位数:   {np.median(arr):.1f} tokens")
    print(f"  最小值:   {arr.min()} tokens")
    print(f"  最大值:   {arr.max()} tokens")
    print(f"  标准差:   {arr.std():.1f}")
    print(f"\n  百分位数:")
    for p in [50, 75, 90, 95, 99]:
        print(f"    {p}%:  {np.percentile(arr, p):.0f} tokens")
    
    # 长度分布
    print(f"\n  长度分布:")
    bins = [0, 256, 512, 1024, 1536, 2048, 3072, 4096, float('inf')]
    bin_labels = ['<256', '256-512', '512-1024', '1024-1536', '1536-2048', '2048-3072', '3072-4096', '>4096']
    for i in range(len(bins)-1):
        count = np.sum((arr >= bins[i]) & (arr < bins[i+1]))
        pct = count / len(arr) * 100
        bar = '█' * int(pct / 2)
        print(f"    {bin_labels[i]:12s}: {count:4d} ({pct:5.1f}%) {bar}")

print_stats("Instruction 长度", instruction_lengths)
print_stats("Response 长度", response_lengths)
print_stats("Total 长度 (Instruction + Response)", total_lengths)

# 推荐max_length
print(f"\n{'='*80}")
print("💡 推荐的 max_seq_length 设置")
print(f"{'='*80}")

total_arr = np.array(total_lengths)
for coverage in [90, 95, 99]:
    percentile_val = np.percentile(total_arr, coverage)
    truncated = np.sum(total_arr > percentile_val)
    print(f"  max_length = {int(percentile_val):4d}  →  覆盖 {coverage}% 样本 (截断 {truncated} 个样本)")

print(f"\n当前设置: max_length = 2048")
covered = np.sum(total_arr <= 2048)
coverage_pct = covered / len(total_arr) * 100
truncated = len(total_arr) - covered
print(f"  ✓ 覆盖 {covered}/{len(total_arr)} 个样本 ({coverage_pct:.1f}%)")
print(f"  ✗ 需要截断 {truncated} 个样本 ({100-coverage_pct:.1f}%)")

if coverage_pct >= 95:
    print(f"\n✅ 当前max_length=2048 已经覆盖95%+样本，设置合理！")
elif coverage_pct >= 90:
    print(f"\n🟡 当前max_length=2048 覆盖90-95%样本，基本够用。")
    recommended = int(np.percentile(total_arr, 95))
    print(f"   如果想覆盖更多，建议增加到 {recommended}")
else:
    print(f"\n🔴 当前max_length=2048 覆盖不足90%样本！")
    recommended = int(np.percentile(total_arr, 95))
    print(f"   建议增加到至少 {recommended} 以覆盖95%样本")

# 如果太多样本被截断
avg_length = total_arr.mean()
if avg_length < 1024:
    print(f"\n💰 优化建议: 平均长度仅{avg_length:.0f}，可以考虑降低max_length到1024或1536来:")
    print(f"   • 节省显存 (可以增大batch_size)")
    print(f"   • 加快训练速度 (减少padding)")

print(f"\n{'='*80}")

