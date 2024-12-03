import torch
from transformers import BertTokenizer, BertModel, BertConfig
import time
import numpy as np

# 加载 BERT 模型和分词器
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")
model.eval()  # 切换到评估模式

# 获取模型配置
config = model.config

# 打印模型的超参数
print(f"Model Hyperparameters:")
print(f"  hidden_size: {config.hidden_size}")
print(f"  num_hidden_layers: {config.num_hidden_layers}")
print(f"  num_attention_heads: {config.num_attention_heads}")
print(f"  intermediate_size: {config.intermediate_size}")
print(f"  max_position_embeddings: {config.max_position_embeddings}")
print(f"  vocab_size: {config.vocab_size}")
print(f"  hidden_dropout_prob: {config.hidden_dropout_prob}")
print(f"  attention_probs_dropout_prob: {config.attention_probs_dropout_prob}")
print(f"  layer_norm_eps: {config.layer_norm_eps}")
print(f"  initializer_range: {config.initializer_range}")
print(f"  type_vocab_size: {config.type_vocab_size}")
print(f"  max_position_embeddings: {config.max_position_embeddings}")
print(f"  num_labels: {config.num_labels}")

# 示例输入文本
input_strings = [
    "how are you! i am very happy to see you guys, please give me five ok? thanks",
    "this is some jokes, please tell somebody else that reputation to user privacy protection. There is no central authority or supervisor having overall manipulations over others, which makes Bitcoin favored by many. Unlike filling piles of identity information sheets before opening bank accounts, users of Bitcoin need only a pseudonym, a.k.a an address or a hashed public key, to participate the system."
]

# 最大序列长度和批量大小
max_seq_len = 128
batch_size = len(input_strings)

# 将文本转换为输入 ID 和注意力掩码
input_ids = []
attention_masks = []
for text in input_strings:
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # 添加[CLS]和[SEP]标记
        max_length=max_seq_len,
        pad_to_max_length=True,  # 填充至最大长度
        truncation=True,
        return_tensors='pt'  # 返回 PyTorch 张量
    )
    input_ids.append(encoding['input_ids'])
    attention_masks.append(encoding['attention_mask'])

# 转换为 PyTorch 张量
input_ids = torch.cat(input_ids, dim=0)  # 批量输入
attention_masks = torch.cat(attention_masks, dim=0)

# 将输入数据发送到 CPU (如果需要)
device = torch.device("cpu")
model.to(device)
input_ids = input_ids.to(device)
attention_masks = attention_masks.to(device)

# 执行推理10次并计算总时间
total_inference_time = 0.0
for _ in range(1):
    start_time = time.time()
    with torch.no_grad():  # 在推理时禁用梯度计算
        outputs = model(input_ids, attention_mask=attention_masks)
    end_time = time.time()

    # 累加每次推理的时间（毫秒）
    inference_time = (end_time - start_time) * 1000  # 转换为毫秒
    total_inference_time += inference_time

# 输出10次推理的总时间
print(f"Total PyTorch BERT inference time for 1 run: {total_inference_time:.4f} ms")
