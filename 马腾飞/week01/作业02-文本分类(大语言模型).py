from openai import OpenAI

client = OpenAI(
    api_key="sk-9c73024ed905491da7a4b414c0adaff7",
    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def text_calssify_using_llm(text: str) -> str:
    """
    文本分类（大语言模型），输入文本完成类别划分
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

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
"""},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content


if __name__ == '__main__':
    print("大语言模型: ", text_calssify_using_llm("帮我导航到天安门"))

    print("大语言模型: ", text_calssify_using_llm("天宫院地铁"))

