import os
from openai import OpenAI

def text_classify(text : str) -> str:
    try:
        client = OpenAI(
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
            # 新加坡和北京地域的API Key不同。获取API Key: https://help.aliyun.com/zh/model-studio/get-api-key
            api_key="sk-4f0918ea000c45faa4274d3f171a8479",
            # 以下是北京地域base_url，如果使用新加坡地域的模型，需要将base_url替换为: https://dashscope-intl.aliyuncs.com/compatible-mode/v1
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen-plus",  # 模型列表: https://help.aliyun.com/zh/model-studio/getting-started/models
            messages=[
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': f"""请帮我分类：{text}, 分类的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
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
                                            只输出类别信息，不要输出别的内容。"""}])


        print(completion.choices[0].message.content)
    except Exception as e:
        print(f"错误信息：{e}")
        print("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")

if __name__ == '__main__':
    text_classify("帮我导航到最近的网吧")