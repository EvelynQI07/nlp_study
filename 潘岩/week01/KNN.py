import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=1000)
print(dataset.head(5))

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sententce.values) # 统计词表
input_feature = vector.transform(input_sententce.values) # 100 * 词表大小

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print(model)

test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
