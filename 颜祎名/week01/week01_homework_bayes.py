import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

#1.加载数据
df = pd.read_csv('dataset.csv', sep='\t', encoding='utf-8', names=['text', 'label'])

#2.中文分词
def chinese_word_cut(text : str) -> str:
    return " ".join(jieba.lcut(str(text)))

df['cut_text'] = df['text'].apply(chinese_word_cut)

#3.划分数据集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(df['cut_text'], df['label'], test_size=0.2, random_state=42)

#4.向量化
vec = CountVectorizer()
x_train_counts = vec.fit_transform(X_train)

#5.训练模块
nb = MultinomialNB()
nb.fit(x_train_counts, y_train)

#6.测试效果
score = nb.score(vec.transform(X_test), y_test)
print(f"通过贝叶斯统计方式训练模型准确率:{score:.4f}")










