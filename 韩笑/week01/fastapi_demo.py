import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN
from openai import OpenAI

from fastapi import FastAPI

#引入fastapi
app = FastAPI()

# 使用pandas加载数据集
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=100000)

# print(dataset.head(5))

# 建立模型
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 通过jieba进行分词使用空格间隔拼接

vector = CountVectorizer()  # 提取特征 词频统计
vector.fit(input_sentence.values)  # 统计词表
input_feature = vector.transform(input_sentence.values)  # nrows*词表大小

model = KNeighborsClassifier()  # 导入模型
model.fit(input_feature, dataset[1].values)  # 根据数据集和提取的特征进行训练模型

# 引入大模型
client = OpenAI(
    api_key="sk-777ae59d8b3e451db4dd91fe6961dbe5",

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


@app.get("/")
def read_root():
    return {"Hello": "你好学习机器学习和大模型的同学！"}

@app.get("/test-cls/ml")
def student_learn_ml(text: str) -> str:
    # 通过模型预测
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]
    """ 直接return model.predict(test_feature)时出现错误 
    错误是因为函数声明的返回类型是 str，但实际返回的是 numpy.ndarray 类型。需要将预测结果转换为字符串
    python语法不熟悉
    """

@app.get("/test-cls/llm")
def student_learn_llm(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen3-coder-plus",  # 模型代号
        messages=[
            {"role": "system", "content": "请问需要什么帮助"},
            {"role": "user", "content": f"""
            帮我进行分类：{text}
            输出类别按照下面的分类
            FilmTele-Play            
            Video-Play               
            Music-Play              
            Radio-Listen           
            Alarm-Update        
            Travel-Query        
            HomeAppliance-Control  
            Weather-Query          
            Calendar-Query      
            TVProgram-Play      
            Audio-Play       
            Other  """},
        ]
    )
    return completion.choices[0].message.content
