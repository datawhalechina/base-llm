import torch
import os
from transformers import AutoTokenizer, AutoModel

# 1. 环境和模型配置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "bert-base-uncased"
texts = ["He likes studying", "我喜欢自然语言处理"]

# 2. 加载模型和分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
model.eval()

print("\n=== 详细分词分析 ===")

# 3. 文本预处理
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)

# 详细分析每个文本的分词结果
for i, text in enumerate(texts):
    print(f"\n--- 文本 {i + 1}: '{text}' ---")

    # 单独对每个文本进行分词，查看详细过程
    encoded = tokenizer.encode_plus(text, add_special_tokens=True)
    tokens = tokenizer.convert_ids_to_tokens(encoded['input_ids'])

    print("完整分词序列:")
    print(" | ".join(tokens))

    # 显示每个token的详细信息
    print("\nToken详细信息:")
    for j, (token_id, token) in enumerate(zip(encoded['input_ids'], tokens)):
        print(f"  位置 {j}: ID={token_id:4d}, Token='{token}'")

        # 检查是否有子词标记
        if token.startswith('##'):
            print(f"      ↑ 这是一个子词标记 (continuation of previous word)")

# 4. 打印整体的tokenizer输出
print("\n--- Tokenizer 完整输出 ---")
for key, value in inputs.items():
    print(f"{key}: \n{value}\n")

# 5. 解码batch中的所有token
print("--- Batch中所有Token的解码 ---")
batch_tokens = []
for i in range(inputs['input_ids'].shape[0]):
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][i])
    batch_tokens.append(tokens)
    print(f"文本 {i}: {tokens}")

# 6. 模型推理
with torch.no_grad():
    outputs = model(**inputs)

# 7. 提取特征
last_hidden_state = outputs.last_hidden_state
sentence_features_pooler = getattr(outputs, "pooler_output", None)
sentence_features = last_hidden_state[:, 0, :]

print("\n--- 特征提取结果 ---")
print(f"句子特征 shape: {sentence_features.shape}")
if sentence_features_pooler is not None:
    print(f"pooler_output shape: {sentence_features_pooler.shape}")

# 8. 特别关注第一个句子的分词
print("\n=== 重点分析: 'He likes studying' ===")
study_tokens = tokenizer.tokenize("studying")
print(f"'studying' 的分词结果: {study_tokens}")

# 检查词汇表中是否有"##ing"
if '##ing' in tokenizer.vocab:
    ing_id = tokenizer.vocab['##ing']
    print(f"找到 '##ing' 在词汇表中，ID: {ing_id}")
else:
    print("在词汇表中没有找到 '##ing'")

# 检查其他可能的子词
test_words = ["playing", "running", "eating", "studying"]
for word in test_words:
    tokens = tokenizer.tokenize(word)
    print(f"'{word}' -> {tokens}")