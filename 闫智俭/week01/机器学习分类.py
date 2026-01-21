import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer #词频统计
from sklearn.neighbors import KNeighborsClassifier #KNN

data = pd.read_csv(filepath_or_buffer="dataset.csv",sep="\t",names=["text","label"],nrows=None)
input_sententce =dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) #sklearn对中文处理

vector = CountVectorizer() #对文本进行提取特征 默认是使用标点符号分词，不是模型
vector.fit(input_sententce.values) #统计词表
input_feature = vector.transform(input_sententce.values) #词表大小

model = KNeighborsClassifier()
model.fit(input_feature,dataset[1].values)

def_text_classify_using_ml(taxt=str) -> str...

def_text_classify_using_llm(taxt=str) -> str:...

if __name__ == '__main__':
    print(text_classify_using_ml("帮我导航到天安门"))