import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN

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

# 通过模型预测
test_query = "帮我播放孙燕姿的歌曲"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果", model.predict(test_feature))
#数据集数量10000时预测结果 KNN模型预测结果 ['FilmTele-Play']
#数据集数量100000时预测结果 KNN模型预测结果 ['Music-Play']