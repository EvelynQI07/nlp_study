import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

dataset = pd.read_csv('dataset.csv', sep='\t', header=None, nrows=10000)
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

def text_classify_using_ml(text: str) -> str:
    """
    通过机器学习方法进行文本分类
    """
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = vector.transform([text_sentence])
    return model.predict(text_feature)

def text_classify_using_llm(text: str) -> str:
    """
    通过大语言模型进行文本分类
    """
    client = OpenAI(
        api_key='sk-xxxxxxxxx',
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    completion = client.chat.completions.create(
        model = "qwen-flash",
        messages=[{"role":"user", "content":
            f"""帮我进行文本分类：{text}
            输出的类别只能从如下中进行选择:
            FiLmTeLe-Play
            Video-Play
            Music-PLay
            Radio-Listen
            ALarm-Update
            TraveL-Query
            HomeAppliance-Control
            Weather-Query
            Calendar-Query
            TVProgram-Play
            Audio-PLay
            Other
            """}]
    )
    return completion.choices[0].message.content

if __name__ == '__main__':
    print(text_classify_using_ml("帮我导航到天安门"))
    print(text_classify_using_llm("帮我导航到天安门"))
