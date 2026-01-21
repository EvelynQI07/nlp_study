import pandas as pd
import jieba
import sklearn
import torch
import transformers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

dataset=pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)


texts = dataset[0]
labels = dataset[1]

texts_cut = texts.apply(
    lambda x: " ".join(jieba.lcut(x))
)

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts_cut)
y = labels.values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2
)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

knn_pred = knn.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)
print("KNN 模型的准确率是：", knn_acc)



nb = MultinomialNB()
nb.fit(X_train, y_train)

nb_pred = nb.predict(X_test)
nb_acc=accuracy_score(y_test,nb_pred)
print("朴素贝叶斯 模型的准确率是",nb_acc)
