from openai import OpenAI

client = OpenAI(
    api_key="sk-7fdec2709521405db619bef9231ebf14",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def text_classify(text:str) -> str:
    completion = client.chat.completions.create(
        model="qwen-flash",
        messages=[
            {
                "role": "user",
                "content": f"""帮我进行文本分类：{text}
                    输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。不需输出额外文本
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