import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

# 读取数据集，使用制表符分隔，无标题行，限制读取10000行
df = pd.read_csv("dataset.csv", sep='\t', header=None, nrows=10000)
# 输出标签列的值计数，查看各类别分布情况
#print(df[1].value_counts())
input_sentence = df[0].apply(lambda x: ' '.join(jieba.lcut(x))) # 分词
vectorizer = CountVectorizer() # 对文本进行提取特征
vectorizer.fit(input_sentence.values) # 统计词表
input_features = vectorizer.transform(input_sentence.values) # 提取特征 进行转换
model = KNeighborsClassifier()
model.fit(input_features, df[1].values)

# 创建OpenAI客户端
client = OpenAI(
    api_key="sk-e2b4434aaf5544c0805d21e13e806e3f",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
# 定义一个函数，用于与AI进行对话
def text_calssify_using_llm(text: str) -> str:
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
"""},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

#机器学习 进行文本分类
def text_classify_using_model(text: str) -> str:
    """
    使用训练好的KNN模型对输入文本进行分类
    参数:
        text (str): 需要分类的文本
    返回:
        str: 预测的类别标签
    """
    # 对输入文本进行分词并重新组合为字符串
    test_sentence = " ".join(jieba.lcut(text))
    # 将分词后的句子转换为特征向量
    test_features = vectorizer.transform([test_sentence])
    # 使用模型进行预测并返回结果
    predicted_label = model.predict(test_features)[0]
    return str(predicted_label)


if __name__ == "__main__":
        reply = text_calssify_using_llm("去年台湾有个都市片，奇幻题材的帮我播放")
        print(f"大语言模型回复：{reply}")
        reply = text_classify_using_model("去年台湾有个都市片，奇幻题材的帮我播放")
        print(f"机器学习模型回复：{reply}")