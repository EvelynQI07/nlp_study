
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import jieba
from collections import Counter
import numpy as np

# ===================== 1. 数据读取与预处理 ======================
# 读取数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=1000)
texts = dataset[0].tolist()  # 文本列表
labels = dataset[1].tolist()  # 标签列表

# 步骤1：分词（jieba）
seg_texts = [" ".join(jieba.lcut(text)) for text in texts]

# 步骤2：构建词汇表（将词语转为数字编码）
all_words = []
for text in seg_texts:
    all_words.extend(text.split())
word_count = Counter(all_words)  # 统计词频
vocab = {word: idx + 1 for idx, (word, _) in enumerate(word_count.most_common(1000))}  # 保留前1000个高频词
vocab["<PAD>"] = 0  # 填充符，用于统一文本长度

# 步骤3：标签编码（将文本标签转为数字）
label2id = {label: idx for idx, label in enumerate(set(labels))}
id2label = {idx: label for label, idx in label2id.items()}
labels_id = [label2id[label] for label in labels]


# 步骤4：文本转数字序列（统一长度为20）
def text2seq(text, vocab, max_len=20):
    words = text.split()
    seq = [vocab.get(word, 0) for word in words[:max_len]]  # 截断
    seq += [0] * (max_len - len(seq))  # 填充
    return seq


texts_seq = [text2seq(text, vocab) for text in seg_texts]


# ===================== 2. 定义Dataset和DataLoader =====================
class TextDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = torch.tensor(texts, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


# 划分训练集和测试集（8:2）
train_size = int(0.8 * len(texts_seq))
train_texts = texts_seq[:train_size]
train_labels = labels_id[:train_size]
test_texts = texts_seq[train_size:]
test_labels = labels_id[train_size:]

# 构建DataLoader（批量加载数据）
train_dataset = TextDataset(train_texts, train_labels)
test_dataset = TextDataset(test_texts, test_labels)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)


# ===================== 3. 定义CNN文本分类模型 =====================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes):
        super(TextCNN, self).__init__()
        # 词嵌入层：将数字编码转为向量
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 卷积层：提取文本局部特征
        self.convs = nn.ModuleList([
            nn.Conv2d(1, 16, (k, embed_dim)) for k in [2, 3, 4]  # 2/3/4元语法卷积
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # 全连接层：分类
        self.fc = nn.Linear(16 * 3, num_classes)

    def forward(self, x):
        # x: [batch_size, max_len]
        x = self.embedding(x)  # [batch_size, max_len, embed_dim]
        x = x.unsqueeze(1)  # [batch_size, 1, max_len, embed_dim]（添加通道维度）
        # 卷积+池化
        outs = []
        for conv in self.convs:
            out = conv(x)  # [batch_size, 16, max_len-k+1, 1]
            out = self.relu(out.squeeze(3))  # [batch_size, 16, max_len-k+1]
            out = nn.MaxPool1d(out.size(2))(out).squeeze(2)  # [batch_size, 16]
            outs.append(out)
        out = torch.cat(outs, dim=1)  # [batch_size, 16*3]
        out = self.dropout(out)
        out = self.fc(out)  # [batch_size, num_classes]
        return out


# 初始化模型
vocab_size = len(vocab)
embed_dim = 32  # 词向量维度
num_classes = len(label2id)  # 分类类别数
model = TextCNN(vocab_size, embed_dim, num_classes)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ===================== 4. 训练模型 =====================
criterion = nn.CrossEntropyLoss()  # 损失函数（分类任务）
optimizer = optim.Adam(model.parameters(), lr=0.001)  # 优化器

# 训练循环
epochs = 10
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for texts_batch, labels_batch in train_loader:
        texts_batch = texts_batch.to(device)
        labels_batch = labels_batch.to(device)

        # 前向传播
        outputs = model(texts_batch)
        loss = criterion(outputs, labels_batch)

        # 反向传播+优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# ===================== 5. 模型评估 =====================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for texts_batch, labels_batch in test_loader:
        texts_batch = texts_batch.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(texts_batch)
        _, predicted = torch.max(outputs.data, 1)

        total += labels_batch.size(0)
        correct += (predicted == labels_batch).sum().item()

print(f"测试集准确率: {100 * correct / total:.2f}%")


# ===================== 6. 单文本预测 =====================
def predict(text):
    model.eval()
    # 预处理
    seg_text = " ".join(jieba.lcut(text))
    seq = text2seq(seg_text, vocab)
    seq = torch.tensor(seq, dtype=torch.long).unsqueeze(0).to(device)

    # 预测
    with torch.no_grad():
        output = model(seq)
        _, pred_id = torch.max(output.data, 1)
    return id2label[pred_id.item()]


# 测试预测
test_query = "帮我播放一下郭德纲的小品"
print(f"待预测文本：{test_query}")
print(f"PyTorch模型预测结果：{predict(test_query)}")
