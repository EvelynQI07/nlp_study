#逻辑回归
import pandas as pd
import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

# #1、准备数据
data = pd.read_csv("dataset.csv",sep = "\t",header = None,nrows= None )
input_sentence = data[0].apply(lambda x :" ".join(jieba.lcut(x)))

# #2将文本向量化
vector = TfidfVectorizer()
X_train = vector.fit_transform(input_sentence)
y_train = data[1].values

#3创建并训练模型
model = LogisticRegression()
model.fit(X_train,y_train)

#4进行预测
new_text = " ".join(jieba.lcut("到天安门的路怎么走？"))
new_x = vector.transform([new_text])
prediction = model.predict(new_x)[0]
print(prediction)

#KNN
import pandas as pd
import jieba
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

# #1、准备数据
data = pd.read_csv("dataset.csv",sep = "\t",header = None,nrows= None )
input_sentence = data[0].apply(lambda x :" ".join(jieba.lcut(x)))

# #2将文本向量化
vector = CountVectorizer()
vector.fit(input_sentence)
X_train = vector.transform(input_sentence)


#3创建并训练模型
model = LogisticRegression()
model.fit(X_train,data[1].values)

#4封装成一个函数
def text_classify_using_lm(input_sentence:str)->str:
    new_text = " ".join(jieba.lcut("input_sentence"))
    new_x = vector.transform([new_text])
    return model.predict(new_x)[0]

#5调用这个函数
print(text_classify_using_lm("放周杰伦的歌"))
