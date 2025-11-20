import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# 1. 加载模型
model_name = "t5-small" # 使用最小版本演示
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

# 2. 准备输入
# T5 需要明确的任务前缀
input_text_1 = "translate English to German: The house is wonderful."
input_text_2 = "stsb sentence1: The rhino grazed on the grass. sentence2: A rhino is grazing in a field."

# 3. 推理生成
inputs = tokenizer([input_text_1, input_text_2], return_tensors="pt", padding=True)
outputs = model.generate(**inputs)

print(f"输入 1: {input_text_1}")
print(f"输出 1: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

print(f"输入 2: {input_text_2}")
print(f"输出 2: {tokenizer.decode(outputs[1], skip_special_tokens=True)}")