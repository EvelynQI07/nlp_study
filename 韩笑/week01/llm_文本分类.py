from openai import OpenAI

# 引入大模型
client = OpenAI(
    api_key="sk-777ae59d8b3e451db4dd91fe6961dbe5",

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    model="qwen3-coder-plus",#模型代号
    messages = [
        {"role": "system", "content": "请问需要什么帮助"},
        {"role": "user", "content": f"""
        帮我进行分类：播放孙燕姿的歌曲
        输出类别按照下面的分类
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
        Other  """},
    ]
)

print(completion.choices[0].message.content)