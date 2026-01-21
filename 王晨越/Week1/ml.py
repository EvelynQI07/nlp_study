import numpy
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from sklearn import neighbors
from sklearn import tree

dataset = pd.read_csv("./dataset.csv", sep="\t", header=None, nrows=1000)
sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(sentence.values)
vector_data = vector.transform(sentence.values)
print(dataset[1].unique())
test_sentence = "帮我导航去北京"

#KNeighbors
for k in [1,3,5,7,9]:
    knModel = neighbors.KNeighborsClassifier(n_neighbors=k)
    knModel.fit(vector_data, dataset[1].values)
    prediction = knModel.predict(vector.transform([" ".join(jieba.lcut(test_sentence))]))
    print(test_sentence)
    print("KNN模型预测结果: "+"(neighbors="+str(k)+")=    "+ prediction)

#Decision tree
treeModel = tree.DecisionTreeClassifier()
treeModel.fit(vector_data, dataset[1].values)
tree_prediction = treeModel.predict(vector.transform([" ".join(jieba.lcut(test_sentence))]))
print(test_sentence)
print("Decision Tree模型预测结果=     "+ tree_prediction)