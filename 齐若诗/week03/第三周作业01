import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

# 1. 数据准备 (Data Preparation)
    try:
        dataset = pd.read_csv("C:/Users/ruosh/nlp/nlp20/Week01/dataset.csv", sep="\t", header=None, nrows=100)
        texts = dataset[0].tolist()
        string_labels = dataset[1].tolist()
        print(f"成功读取本地数据，共 {len(texts)} 条。")
    except FileNotFoundError:
        print("Warning: 未找到 dataset.csv，正在生成模拟数据以供演示...")
        # 生成一些简单的分类任务数据
        texts = [
                    "帮我导航到北京", "我要去上海", "定位到深圳", "导航去广州",  # 导航
                    "明天天气怎么样", "查询北京天气", "下雨了吗", "广州气温多少",  # 天气
                    "播放周杰伦的歌", "我想听音乐", "放一首这是我的", "停止播放"  # 音乐
                ] * 50  # 复制50次增加数据量
        string_labels = (["导航"] * 4 + ["天气"] * 4 + ["音乐"] * 4) * 50

    # 标签编码
    label_to_index = {label: i for i, label in enumerate(set(string_labels))}
    index_to_label = {i: label for label, i in label_to_index.items()}
    numerical_labels = [label_to_index[label] for label in string_labels]

    # 字符编码 (构建词表)
    char_to_index = {'<pad>': 0}
    for text in texts:
        for char in text:
            if char not in char_to_index:
                char_to_index[char] = len(char_to_index)

    vocab_size = len(char_to_index)
    max_len = 20
    # Dataset 定义
    class TextDataset(Dataset):
        def __init__(self, texts, labels, char_to_index, max_len):
            self.texts = texts
            self.labels = torch.tensor(labels, dtype=torch.long)
            self.char_to_index = char_to_index
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, idx):
            text = self.texts[idx]
            indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            indices += [0] * (self.max_len - len(indices))  # Padding
            return torch.tensor(indices, dtype=torch.long), self.labels[idx]


    dataset_obj = TextDataset(texts, numerical_labels, char_to_index, max_len)
    dataloader = DataLoader(dataset_obj, batch_size=16, shuffle=True)


# 2. 核心模型：RNN/LSTM/GRU

    class UniversalRNN(nn.Module):
        def __init__(self, model_type, vocab_size, embedding_dim, hidden_dim, output_dim):
            super(UniversalRNN, self).__init__()
            self.model_type = model_type
            self.embedding = nn.Embedding(vocab_size, embedding_dim)

            # 官方中：PyTorch 的 RNN/LSTM/GRU 参数几乎一致
            # batch_first=True 意味着输入格式为 (Batch, Seq_Len, Feature)
            if model_type == 'RNN':
                self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
            elif model_type == 'LSTM':
                self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
            elif model_type == 'GRU':
                self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
            else:
                raise ValueError("不支持的模型类型，请选择 RNN, LSTM 或 GRU")

            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # 1. Embedding
            # x: [batch, seq_len] -> [batch, seq_len, embed_dim]
            embedded = self.embedding(x)

            # 2. RNN Layer
            # 区别处理：LSTM 返回 (output, (h, c))，而 RNN/GRU 返回 (output, h)
            if self.model_type == 'LSTM':
                output, (hidden, cell) = self.rnn(embedded)
            else:
                output, hidden = self.rnn(embedded)

            # hidden shape: [num_layers * num_directions, batch, hidden_dim]
            # 只需要最后一层的隐藏状态
            last_hidden = hidden[-1]

            # 3. Fully Connected
            out = self.fc(last_hidden)
            return out


# 3. 训练与对比实验逻辑
    def run_experiment(model_type, epochs=15):
        print(f"\n[{model_type}] 开始训练...")

        # 超参数
        embedding_dim = 64
        hidden_dim = 128
        output_dim = len(label_to_index)

        model = UniversalRNN(model_type, vocab_size, embedding_dim, hidden_dim, output_dim)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.002)  # 稍微调大学习率

        start_time = time.time()
        history_acc = []

        for epoch in range(epochs):
            model.train()
            correct = 0
            total = 0
            total_loss = 0

            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            history_acc.append(acc)

            # 简单的进度打印
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(dataloader):.4f} | Acc: {acc:.2f}%")

        duration = time.time() - start_time
        print(f"[{model_type}] 训练结束. 耗时: {duration:.2f}s, 最终精度: {history_acc[-1]:.2f}%")

        return history_acc, duration, model


# 4. 执行对比
    models = ['RNN', 'LSTM', 'GRU']
    results = {}

    for m in models:
        acc_history, duration, model_instance = run_experiment(m, epochs=20)
        results[m] = {
            'final_acc': acc_history[-1],
            'duration': duration,
            'history': acc_history,
            'model': model_instance  # 保存模型以便测试
        }

# 5. 结果汇总与测试
    print("\n" + "=" * 50)
    print(f"{'模型':<10} | {'最终精度':<15} | {'训练耗时':<15}")
    print("-" * 50)
    for m in models:
        print(f"{m:<10} | {results[m]['final_acc']:.2f}%          | {results[m]['duration']:.2f}s")
    print("=" * 50)


    # 简单测试预测
    def predict(text, model, char_to_index, max_len):
        model.eval()
        indices = [char_to_index.get(char, 0) for char in text[:max_len]]
        indices += [0] * (max_len - len(indices))
        input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            out = model(input_tensor)
            _, pred = torch.max(out, 1)
        return index_to_label[pred.item()]


    test_text = "查询北京天气"
    print(f"\n测试输入: '{test_text}'")
    for m in models:
        pred_label = predict(test_text, results[m]['model'], char_to_index, max_len)
        print(f"{m} 预测结果: {pred_label}")
