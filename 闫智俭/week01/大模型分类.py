import pandas as pd


def text_classify_using_ml(text:str) -> str:



    pass


def text_classify_using_llm(text:str) -> str:



    completion = client.chat.completions.create
        model="qwen-flash",

        message=
            {"role":"user","content":"""帮我进行文本分类：{text} 
输出的类别只能从如下中选择
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
    return completion.choices{0}.message.content
    pass

if __name__ == '__main__':
    print("2222")

    print("机器学习:",text_classify_using_ml("帮我导航到天安门"))
    print("大语言模型:",text_classify_using_llm("帮我导航到天安门"))

