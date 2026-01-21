# import jieba
# import pandas as pd
#
# from sklearn.feature_extraction.text import CountVectorizer
#
# from sklearn.neighbors import KNeighborsClassifier
#
# dataset = pd.read_csv("./dataset.csv", sep="\t", header=None, nrows=100)
#
# print(dataset.head(5))
#
# input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
#
# vector = CountVectorizer()
# vector.fit(input_sentence.values)
# input_feature = vector.transform(input_sentence)
#
# model = KNeighborsClassifier()
# model.fit(input_feature, dataset[1].values)
# print(model)
#
# test_query="帮我播放一下郭德纲的小品"
# test_sentence = " ".join(jieba.lcut(test_query))
# test_feature = vector.transform([test_sentence])
# print("待预测的文本", test_query)
# print("KNN模型预测结果: ", model.predict(test_feature))


import pandas as pd
import jieba

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

from openai import OpenAI

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)
print(dataset[1].value_counts())

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = CountVectorizer()
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

client = OpenAI(
    api_key="sk-026bc9ed889740aa948ebf5b06ebbe99",

    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def text_classify_using_ml(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)

def text_classify_using_llm(text: str) -> str:
    completion = client.chat.conpletions.create(
        model='qwen-flash',

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
FilmTele-Play            
Video-Play               
Music-Play              
Radio-Listen           
Alarm-Update        
Travel-Query        
HomeAppliance-Control  
Weather-Query          
Calendar-Query      
TVProgram-Play      
Audio-Play       
Other             
"""}
        ]
    )
    return completion.choices[0].message.content

if __name__=="__main__":
    print("222")

    print("机器学习： ", text_classify_using_ml("帮我导航到天安门"))
    print("大语言模型：", text_classify_using_ml("帮我导航到天安门"))





