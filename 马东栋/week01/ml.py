import pandas as pd
import jieba as jb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer

def text_preprocess(text: str) -> str:
    """
        对中文句子预处理，分词之后两个词之间加空格
    """
    return " ".join(jb.lcut(text))

def text_feature_extraction(text: str) -> str:
    """
        对输入的文本进行特征提取
    """
    test_sentence = text_preprocess(text)
    return vector.transform([test_sentence])


dataset = pd.read_csv("../dataset.csv",sep = "\t",header= None, nrows=10000)
print(dataset.head(5))

x_train = dataset[0].apply(text_preprocess)
y_train = dataset[1]
print(x_train)

# 对文本进行特征提取
vector = CountVectorizer()
vector.fit(x_train.values)
x_train_feature = vector.transform(x_train.values)
print(x_train_feature)

estimator = KNeighborsClassifier()
estimator.fit(x_train_feature, y_train)
print(estimator)

test_query = "帮我导航到最近的网吧"
test_feature = text_feature_extraction(test_query)
print("预测结果: ", estimator.predict(test_feature))



