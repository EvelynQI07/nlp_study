import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计
from sklearn.neighbors import KNeighborsClassifier  # KNN
from openai import OpenAI
from fastapi import FastAPI

app = FastAPI()
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
client = OpenAI(
    api_key="sk-c6625bb19dc448a7ab54d45902d6b5e3",  # 账号绑定，用来计费的
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


@app.get("/test/knn")
def ml(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]


@app.get("/test/ai")
def ai(text: str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",

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
    """},  
        ]
    )
    return completion.choices[0].message.content
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
