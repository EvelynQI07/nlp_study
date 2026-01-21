import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN
from sklearn.naive_bayes import MultinomialNB  # 朴素贝叶斯（第二种机器学习模型）
from sklearn.model_selection import train_test_split  # 划分训练/测试集
from sklearn.metrics import accuracy_score  # 评估准确率
from openai import OpenAI

# ---------------------- 1. 数据加载与预处理 ----------------------
# 加载数据集（指定列名，方便后续操作）
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
dataset.columns = ["text-action", "label"]  # 命名列：文本、标签

# 查看标签分布（了解数据）
print("标签分布：")
print(dataset["label"].value_counts())
print("-" * 50)

# ---------------------- 2. 数据预处理 (jieba 中文分词)----------------------
'''
jieba 分词会将一个中文字符串分为 数个中文词，下方代码是空格作为间隔
'''
dataset["cut_text"] = dataset["text-action"].apply(lambda x: " ".join(jieba.lcut(x)))
print("文本预处理完成！", dataset.head())
print("-" * 50)
# ---------------------- 3. 数据集划分 ----------------------
# 使用train_test_split划分训练集和测试集，test_size=0.25表示测试集占25%，训练集占75%
# random_state=42确保每次划分结果一致，便于复现实验结果
# 返回四个参数：
# X_train: 训练集的特征数据（分词后的文本）
# X_test: 测试集的特征数据（分词后的文本）
# y_train: 训练集的标签数据（对应的分类标签）
# y_test: 测试集的标签数据（对应的分类标签）
X_train, X_test, y_train, y_test = train_test_split(
    dataset["cut_text"],      # 特征数据：经过jieba分词处理后的文本
    dataset["label"],         # 标签数据：对应的分类标签
    test_size=0.25,          # 测试集占比25%
    random_state=42          # 随机种子，保证结果可复现
)

# 打印各数据集的形状，便于了解划分情况
print(f"训练集特征数量: {X_train.shape[0]}, 测试集特征数量: {X_test.shape[0]}")
print(f"训练集标签数量: {y_train.shape[0]}, 测试集标签数量: {y_test.shape[0]}")
print("训练集和测试集分布：")
print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
print("\n训练集特征样例：")
for i in range(min(3, len(X_train))):  # 显示前3个样本
    print(f"样本 {i+1}: '{X_train.iloc[i]}' -> 标签: {y_train.iloc[i]}")
print("-" * 50)

# ---------------------- 4. 文本特征提取 ----------------------
# 词频向量化（将文本转为数值特征）
vector = CountVectorizer()
# fit_transform() = fit() + transform()
# fit()：计算训练数据的词汇表(vocabulary)和统计信息，在CountVectorizer中主要是构建词汇表，词表的格式是列为单词，行为句子，每个单元表示该句子的词频
# transform()：根据已构建的词汇表（上行中的fit方法）将文本转换为向量，在测试时或者使用模型时使用，因为我们只需要提取文本的特征，而不需要统计所有的标签数据
# fit_transform()：同时执行fit和transform两个操作，先构建词汇表，再转换文本为向量
X_train_vec = vector.fit_transform(X_train)
X_test_vec = vector.transform(X_test)  # 测试集仅转换成特征向量

# ---------------------- 5. 模型1：KNN分类器 ----------------------
# 现在还没有搞懂，只知道这么用
knn_model = KNeighborsClassifier()
knn_model.fit(X_train_vec, y_train) # X_train_vec: 训练集的特征向量, y_train: 训练集的标签向量

# 测试KNN模型
y_pred_knn = knn_model.predict(X_test_vec) # 基于训练集的特征向量，预测标签
knn_acc = accuracy_score(y_test, y_pred_knn) # 预测标签和真实标签之间对比，计算准确率
print(f"KNN模型测试准确率：{knn_acc:.4f}")
print("-" * 50)

# ---------------------- 6. 模型2：朴素贝叶斯分类器（第二种机器学习模型） ----------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)

# 测试朴素贝叶斯模型
y_pred_nb = nb_model.predict(X_test_vec)
nb_acc = accuracy_score(y_test, y_pred_nb)
print(f"朴素贝叶斯模型测试准确率：{nb_acc:.4f}")
print("-" * 50)

# ---------------------- 7. 大语言模型（LLM）分类 ----------------------
client = OpenAI(
    api_key="sk-22b99b6ad9d247cfa01abc81e2fe54e4", # 自己的API Key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

all_labels = dataset["label"].unique().tolist()
labels_str = "\n".join(all_labels)

def text_classify_using_llm(text: str, labels_str: str) -> str:
    """
    使用大语言模型（通义千问）进行文本分类
    """

    try:
        completion = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {"role": "user", "content": f"""严格按照要求完成文本分类任务：
1. 待分类文本：{text}
2. 可选类别列表：
{labels_str}
3. 输出要求：仅输出最匹配的类别名称，不添加任何其他符号。"""},
            ],
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"LLM分类出错：{str(e)}"


# ---------------------- 6. 测试示例 ----------------------
def text_classify_using_ml(text: str, model, vector) -> str:
    """
    通用机器学习分类函数
    """
    cut_text = " ".join(jieba.lcut(text))
    text_vec = vector.transform([cut_text])
    return model.predict(text_vec)[0]


if __name__ == "__main__":
    # 测试示例文本
    test_texts = [
        "帮我导航到天安门",
        "播放周杰伦的稻香",
        "明天北京的天气怎么样",
        "设置早上7点的闹钟"
    ]

    print("=== 文本分类测试结果 ===")
    for text in test_texts:
        knn_result = text_classify_using_ml(text, knn_model, vector)
        nb_result = text_classify_using_ml(text, nb_model, vector)
        llm_result = text_classify_using_llm(text, labels_str)

        print(f"\n测试文本：{text}")
        print(f"KNN分类结果：{knn_result}")
        print(f"朴素贝叶斯分类结果：{nb_result}")
        print(f"LLM分类结果：{llm_result}")

'''
运行结果：
=== 文本分类测试结果 ===

测试文本：帮我导航到天安门
KNN分类结果：Travel-Query
朴素贝叶斯分类结果：Travel-Query
LLM分类结果：Travel-Query

测试文本：播放周杰伦的稻香
KNN分类结果：Music-Play
朴素贝叶斯分类结果：Music-Play
LLM分类结果：Music-Play

测试文本：明天北京的天气怎么样
KNN分类结果：Weather-Query
朴素贝叶斯分类结果：Weather-Query
LLM分类结果：Weather-Query

测试文本：设置早上7点的闹钟
KNN分类结果：Alarm-Update
朴素贝叶斯分类结果：Alarm-Update
LLM分类结果：Alarm-Update
'''