# 1.调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
# 修改后的 SimpleClassifier
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. 数据加载与预处理 ---
dataset = pd.read_csv("C:/Users/ruosh/nlp/nlp20/Week01/dataset.csv", sep="\t", header=None, nrows=100)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40


# --- 2. 自定义数据集---
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


# --- 3. 定义可配置结构的分类器 ---
class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SimpleClassifier, self).__init__()
        self.num_layers = num_layers
        self.relu = nn.ReLU()

        # 统一使用 nn.ModuleList 管理动态层数
        self.layers = nn.ModuleList()
        # 输入层
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 中间隐藏层 (如果层数 > 2)
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 输出层
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = self.relu(layer(x))
        return self.output_layer(x)


# --- 4. 预测函数  ---
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))
    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(bow_vector)
    _, predicted_index = torch.max(output, 1)
    return index_to_label[predicted_index.item()]


# --- 5. 对比实验运行 ---
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=16, shuffle=True)

# 实验配置列表
experiments = [
    {"name": "实验1: 基础模型 (128节点, 2层)", "hidden": 128, "layers": 2},
    {"name": "实验2: 加宽模型 (512节点, 2层)", "hidden": 512, "layers": 2},
    {"name": "实验3: 加深模型 (128节点, 4层)", "hidden": 128, "layers": 4}
]

print(f"{'实验名称':<30} | {'初始 Loss':<10} | {'最终 Loss':<10} | {'预测(导航)'}")
print("-" * 85)

for exp in experiments:
    model = SimpleClassifier(vocab_size, exp["hidden"], len(label_to_index), exp["layers"])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02)  # 稍微调大学习率便于观察对比

    initial_loss = 0
    final_loss = 0

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        if epoch == 0: initial_loss = avg_loss
        final_loss = avg_loss

    # 验证具体文本
    pred_res = classify_text("帮我导航到北京", model, char_to_index, vocab_size, max_len, index_to_label)

    print(f"{exp['name']:<30} | {initial_loss:.4f}    | {final_loss:.4f}    | {pred_res}")
