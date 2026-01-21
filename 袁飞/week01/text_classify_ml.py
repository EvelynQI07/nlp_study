import warnings
# 抑制 pkg_resources 弃用警告（来自 jieba 库内部）
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')

import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN



#文本读取
dataset = pd.read_csv('./dataset.csv', sep="\t", header=None, nrows=100)

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sentence.values) # 统计词表
input_feature = vector.transform(input_sentence.values) # 100 * 词表大小
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def text_classify(text:str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]