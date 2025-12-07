import torch
from transformers import pipeline

model_id = "models/Llama-3.2-3B"

pipe = pipeline(
    "text-generation", 
    model=model_id,
    max_new_tokens=512,
    torch_dtype=torch.bfloat16, 
    device_map="auto",
    return_full_text=False,
    temperature=0.7,
    top_p=0.9,
)

prompts = [
    "What is 15 + 27 = ?",
    "A pen costs as much as a pencil and eraser combined. A pencil costs $1.20 and an eraser costs $0.30. How much will 8 pens cost?",
]

for prompt in prompts:
    print(pipe(prompt)[0]['generated_text'])
    print("-" * 100)