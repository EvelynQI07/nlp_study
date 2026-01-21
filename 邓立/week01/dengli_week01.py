import pandas as pd
import jieba # 中文分词用途
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN

import os
from openai import OpenAI
# 我找的魔塔社区api key
import anthropic

client = anthropic.Anthropic(
    api_key="ms-9e8811d5-7a39-xxxxxxxxxxxxxxxxxxxx",
    base_url="https://api-inference.modelscope.cn",
)

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def text_classify_ml(text: str) -> str:
    """
    文本分类（机器学习）
    """
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

def text_classify_llm(text: str) -> str:
    """
    文本分类（大语言模型）
    """
    completion = client.messages.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        max_tokens=1000,
        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
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
Other             
"""}
        ]
    )
    return completion.content[0].text

if __name__ == '__main__':
    print(text_classify_llm("帮我导航"))
