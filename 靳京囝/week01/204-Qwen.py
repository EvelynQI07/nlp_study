import os

import pandas as pd
from openai import OpenAI

base_char = "-" * 50
def print_line(context: str) -> None:
    print(f"\n\n{base_char} {context} {base_char}")

# 1.加载数据
print_line("1.加载数据")
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

# 2.初始化client
print_line("2.初始化client")
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-054668d7012c4e26bc560e1d8d51198c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
print(client)

# 3.调用大模型推理并返回推理结果
print_line("3.调用大模型推理并返回推理结果")
def predict_category(text: str) -> None:
    completion = client.chat.completions.create(
        model="qwen3-max",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"""请你帮我进行文本分类 {text}
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
        ],
        stream=True
    )
    # 输出结果
    for chunk in completion:
        print(chunk.choices[0].delta.content, end="", flush=True)

# 4.预测结果
print_line("4.预测结果")
text = "播放一个音乐类广播"
print(f"待预测的文本：{text}")
print("使用大模型推理的结果是：")
predict_category(text)