import pandas as pd
import random
import os

# 1. 定义数据生成规则
data_samples = {
    "科技": [
        "苹果发布iPhone17，搭载A17芯片性能爆炸", "华为Mate60 Pro麒麟芯片回归引发热议",
        "OpenAI发布GPT-4，人工智能再上新台阶", "英伟达显卡价格持续上涨，算力需求旺盛",
        "马斯克SpaceX星舰发射失败，但取得部分数据", "小米汽车最新谍照曝光，雷军亲自试驾",
        "微软Copilot全面接入Windows系统", "半导体行业迎来寒冬，三星库存积压严重"
    ],
    "体育": [
        "湖人队詹姆斯砍下40分，带领球队逆转", "梅西获得第八座金球奖，历史第一人",
        "中国女排世界联赛击败巴西，晋级决赛", "曼联主场惨败，主教练滕哈格面临下课",
        "F1红牛车队维斯塔潘提前锁定年度总冠军", "姚明辞去篮协主席职务，引发外界关注",
        "全红婵跳水再现水花消失术，夺得金牌", "C罗在沙特联赛上演帽子戏法"
    ],
    "娱乐": [
        "霉霉泰勒斯威夫特演唱会门票秒空", "诺兰新片《奥本海默》横扫奥斯卡",
        "周杰伦新专辑发布，服务器一度崩溃", "某顶流明星塌房，品牌方紧急解约",
        "春节档电影票房突破100亿，贾玲新片领跑", "BLACKPINK续约存疑，股价应声下跌",
        "奥斯卡最佳影片揭晓，冷门佳作爆冷获奖", "流浪地球3宣布定档，吴京回归主演"
    ],
    "财经": [
        "美联储宣布加息25个基点，美股三大指数下跌", "贵州茅台股价创历史新高，分红方案公布",
        "国际金价持续走高，大妈排队抢购黄金", "比特币跌破3万美元关口，币圈一片哀嚎",
        "恒大地产债务重组失败，许家印被采取措施", "A股再次打响3000点保卫战",
        "CPI数据出炉，通胀压力依然存在", "巴菲特减持比亚迪股份，套现数亿港元"
    ]
}

# 2. 生成 CSV 文件
rows = []
# 生成 200 条数据 (通过随机重复采样模拟)
for _ in range(200):
    label = random.choice(list(data_samples.keys()))
    text = random.choice(data_samples[label])
    rows.append([text, label])

# 保存为 dataset.csv，使用制表符 \t 分隔，无表头
df = pd.DataFrame(rows)
df.to_csv("dataset.csv", sep="\t", header=False, index=False)

print(f"✅ 数据集已生成：dataset.csv，共 {len(df)} 条数据")
print("前5条数据预览：")
print(df.head())

import pandas as pd
import torch
import numpy as np  # 补充缺失的 numpy
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# -------------------------- 1. 数据准备 --------------------------
print("正在加载数据集...")
# 加载刚才生成的 dataset.csv
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
dataset.columns = ["text", "label"]  # 给列起个名字方便操作

# 初始化并拟合标签编码器
lbl = LabelEncoder()
dataset['label_id'] = lbl.fit_transform(dataset['label'])

# 打印类别映射关系，方便后续验证
label_map = {index: label for index, label in enumerate(lbl.classes_)}
print(f"类别映射: {label_map}")

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(
    dataset['text'].values,
    dataset['label_id'].values,
    test_size=0.2,
    stratify=dataset['label_id'].values,
    random_state=42
)

# -------------------------- 模型路径配置 --------------------------
# 优先尝试使用 HuggingFace 在线模型，如果想用本地模型，请修改 model_path
model_path = 'bert-base-chinese'

print(f"正在加载 BERT 模型: {model_path} ...")
try:
    tokenizer = BertTokenizer.from_pretrained(model_path)
    # 自动计算类别数量：len(lbl.classes_) 应该是 4
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(lbl.classes_))
    print("🚀 模型和分词器加载成功！")
except Exception as e:
    print(f"❌ 模型加载失败，请检查网络或路径。错误信息: {e}")
    exit()

# 编码数据
train_encoding = tokenizer(list(x_train), truncation=True, padding=True, max_length=64)
test_encoding = tokenizer(list(x_test), truncation=True, padding=True, max_length=64)


# -------------------------- 2. 数据集和加载器 --------------------------
class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = NewsDataset(train_encoding, y_train)
test_dataset = NewsDataset(test_encoding, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# -------------------------- 3. 训练配置 --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"使用设备: {device}")
optimizer = AdamW(model.parameters(), lr=2e-5)


# 精度计算函数
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


# -------------------------- 4. 训练与验证逻辑 --------------------------
def train(epoch):
    model.train()
    total_train_loss = 0

    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 2 == 0 and step > 0:
            print(f"  Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

    avg_train_loss = total_train_loss / len(train_loader)
    print(f"Epoch {epoch} 完成 | 平均 Loss: {avg_train_loss:.4f}")


def validation():
    model.eval()
    total_eval_accuracy = 0
    total_eval_loss = 0

    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)

        loss = outputs.loss
        logits = outputs.logits

        total_eval_loss += loss.item()

        logits = logits.detach().cpu().numpy()
        label_ids = labels.to('cpu').numpy()
        total_eval_accuracy += flat_accuracy(logits, label_ids)

    print(f"✅ 验证集准确率: {total_eval_accuracy / len(test_dataloader):.4f}")
    print("-" * 30)


# -------------------------- 5. 开始训练 --------------------------
epochs = 3
for epoch in range(epochs):
    train(epoch)
    validation()

print("训练结束！")


# -------------------------- 6. 预测新样本 --------------------------
def predict_sentence(text):
    # 1. 处理文本
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 2. 模型推理
        outputs = model(**inputs)
        logits = outputs.logits

        # 3. 获取最大概率的索引
        pred_id = torch.argmax(logits, dim=1).item()

        # 4. 转换回文字标签
        pred_label = label_map[pred_id]
        return pred_label


print("\n========== 最终测试 ==========")
test_sentences = [
    "OpenAI发布了最新的Sora大模型，视频生成效果惊人",  # 预期：科技
    "今天的A股简直没法看，又跌破了3000点",  # 预期：财经
    "湖人队今天加时赛绝杀对手",  # 预期：体育
    "那部新上映的电影票房已经破亿了"  # 预期：娱乐
]

for text in test_sentences:
    result = predict_sentence(text)
    print(f"文本: {text}")
    print(f"预测: 【{result}】\n")
