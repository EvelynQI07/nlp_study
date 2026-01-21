import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#1.加载数据
df = pd.read_csv('dataset.csv', sep='\t', encoding='utf-8', names=['text', 'label'])

#2.中文分词
def chinese_word_cut(text : str) -> str:
    return " ".join(jieba.lcut(str(text)))

df['cut_text'] = df['text'].apply(chinese_word_cut)

#3.划分数据集 (80% 训练, 20% 测试)
X_train, X_test, y_train, y_test = train_test_split(df['cut_text'], df['label'], test_size=0.2, random_state=42)

#4.使用TF-IDF 向量化 相比 CountVectorizer更好
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)

#5.使用逻辑回归
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)

#6.测试效果
score = lr.score(tfidf.transform(X_test), y_test)
print(f"通过逻辑回归方式训练模型准确率:{score:.4f}")










