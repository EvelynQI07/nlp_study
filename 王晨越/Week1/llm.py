from openai import OpenAI
client = OpenAI(
    #Please do not share to other students
    api_key="sk-9f06aac1b31541958699954fe1ca8432",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

response = client.chat.completions.create(
    model="qwen3-max",
    messages=[
        {"role": "system", "content": """
        请帮我把具体分类，分成以下几种类型中的一个:
        'Travel-Query'
        'Music-Play'
        'FilmTele-Play'
        'Video-Play'
        'Radio-Listen'
        'HomeAppliance-Control'
        'Weather-Query'
        'Alarm-Update'
        'Calendar-Query'
        'TVProgram-Play'
        'Audio-Play'
        """},
        {"role": "user", "content": "帮我导航去北京"}
    ]
)
print(response.choices[0].message.content)