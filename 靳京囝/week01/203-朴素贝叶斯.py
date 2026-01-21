import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.naive_bayes import MultinomialNB # KNN

base_char = "-" * 50
def print_line(context: str) -> None:
    print(f"\n\n{base_char} {context} {base_char}")

# 1.加载数据
print_line("1.加载数据")
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

# 2.文本预处理和特征提取
print_line("2.文本预处理和特征提取")
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
print(input_sentence)

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词， 不是模型
vector.fit(input_sentence.values) # 统计词表
input_feature = vector.transform(input_sentence.values) # 100 * 词表大小
print(input_feature)

# 3.多项式朴素贝叶斯模型
print_line("3.多项式朴素贝叶斯模型")
model = MultinomialNB()  # 多项式朴素贝叶斯，特别适合文本分类
model.fit(input_feature, dataset[1].values)
print(model)

# 4.预测新样本
print_line("4.预测新样本")
test_query = "播放一个音乐类广播" # Radio-Listen
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("朴素贝叶斯预测结果: ", model.predict(test_feature)) # Radio-Listen
