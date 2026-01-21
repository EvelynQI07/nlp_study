import jieba
import pandas as pd
from fastapi import FastAPI
from openai import OpenAI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 1、数据集划分
dataset = pd.read_csv('dataset.csv', sep='\t', names=['text', 'label'])
# 测试、训练集比例为2:8，随机种子为42，确保结果可复现，按类别分层采样
train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42, stratify=dataset['label'])

# 2、jieba进行分词
input_sentence = train_set['text'].apply(lambda text: ' '.join(jieba.lcut(text)))

# 3、词频统计
vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

# 4、训练模型，使用KNN
model = KNeighborsClassifier()
model.fit(input_feature, train_set.label)

# 5、使用测试集验证模型训练准确度
test_sentence = test_set['text'].apply(lambda text: ' '.join(jieba.lcut(text)))  # 分词
test_feature = vector.transform(test_sentence.values)  # 向量化
prediction = model.predict(test_feature)  # 预测结果
label_true = test_set['label'].values  # 获取真实标签
accuracy = accuracy_score(label_true, prediction)  # 计算准确率
print(f"机器学习模型KNN已训练完成，在测试集上的准确率为：{accuracy:.4f}")

app = FastAPI()


@app.get("/text-cls/ml")
def text_classify_using_ml(text: str) -> str:
    test_sentence = ' '.join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    prediction = model.predict(test_feature)[0]
    return prediction


client = OpenAI(api_key="sk-026bc9ed889740aa948ebf5b06ebbe99",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


@app.get("/text-cls/llm")
def text_classify_using_llm(text: str) -> str:
    # 调用 LLM 接口：通过 client 对象（通常是 OpenAI 或类似 API 客户端）发起聊天补全请求：
    # completion → 存储模型返回的完整响应对象。
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号
        messages=[
            {"role": "user",  # 表明为用户输入
             "content": f"""帮我进行文本分类：{text}，请在以下的类别中选择最合适的类别：
                FilmTele-Play、Video-Play、Music-Play、Radio-Listen、Alarm-Update、Travel-Query、HomeAppliance-Control、
                Weather-Query、Calendar-Query、TVProgram-Play、Audio-Play、Other"""
             }  # 用户的提问
        ]
    )
    return completion.choices[0].message.content
