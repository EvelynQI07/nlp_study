# 第一种方式：用 sklearn 机器学习方式

import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer as CV # 词频统计
from sklearn.neighbors import KNeighborsClassifier as KNC

dataset = pd.read_csv('./dataset.csv', sep='\t',header=None)
# print(dataset.head(6))

# 对数据集进行预处理
input_sentence = dataset[0].apply(lambda x:' '.join(jieba.lcut(x)))
# print(input_sentence)

vector = CV()
vector.fit(input_sentence.values) # 预处理
input_feature = vector.transform(input_sentence.values) # 转换特征
# print(input_feature)

knc_model = KNC()
knc_model.fit(input_feature, dataset[1].values)  # 模型匹配类别

input_query = input('请输入问题描述：')
q_str = ' ' . join(jieba.lcut(input_query))
q_feature = vector.transform([q_str])     # 注意这里要传列表
q_match = knc_model.predict(q_feature)
print('特征是：', q_feature)
print('类型匹配结果：', q_match)

#------------------------------------------------------------------------------------------------------------------------------------------------

# 第二种方式   用 llm 大语言模型方式
import os
from openai import OpenAI
import pandas as pd

client = OpenAI(
    api_key = 'sk-xxxxx',
    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
)

# fh1 = open('./dataset.csv', 'r')
# dataset = ''
# while True:
#     data = fh1.read(1024)
#     if len(data) == 0:
#         break
#     dataset += data
dataset = pd.read_csv('./dataset.csv', sep='\t', header=None)
typeStr = '\n' . join(dataset[1].values)
# print(typeStr)
# exit()

queryStr = input('请输入提示词：')

completion = client.chat.completions.create(
    # 模型列表
    model = 'qwen-flash',

    # 对话列表
    messages = [
        {"role":"user", "content":f"""
        请帮我分类文本：{queryStr}
        
类型只能从以下这些分类中选择：
{typeStr}
        """},
    ]
)

res = completion.choices[0].message.content

print(res)


