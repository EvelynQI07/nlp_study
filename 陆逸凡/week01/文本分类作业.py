from xml.parsers.expat import model  # XML解析器（这里可能未实际使用）
import jieba  # 中文分词库
import pandas as pd  # 数据处理库
from sklearn import linear_model  # 线性模型模块（逻辑回归等）
from sklearn import tree  # 决策树模块
from sklearn.feature_extraction.text import CountVectorizer  # 词频统计（这里导入但未使用）
from sklearn.neighbors import KNeighborsClassifier  # K近邻分类器
from openai import OpenAI  # 调用大语言模型的API
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF特征提取器

# 1. 读取数据集
# 从指定路径读取CSV文件，使用制表符分隔，无表头，读取前4000行
dataset = pd.read_csv("dataset.csv",   #这里记得改成绝对路径
                      sep="\t", 
                      header=None, 
                      nrows=4000)

# 2. 文本预处理
# 对数据集的第一列（文本数据）进行分词处理
# 使用jieba分词后，用空格连接分词结果，转换为sklearn可处理的格式
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 3. 特征提取（使用TF-IDF方法）
# 创建TF-IDF向量化器，设置参数：
# max_features=1000: 限制特征数量为1000个最常出现的词
# ngram_range=(1, 2): 同时提取单个词和两个词的组合
# min_df=2: 忽略出现次数少于2次的词
# max_df=0.9: 忽略出现频率高于90%的词（去除常见词）
vector = TfidfVectorizer(
    max_features=1000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.9
)

# 4. 训练TF-IDF向量化器
# 在全部训练数据上拟合向量化器，构建词表
vector.fit(input_sententce.values)

# 5. 转换文本数据为特征矩阵
# 将文本数据转换为稀疏特征矩阵，维度为4000×词表大小
input_feature = vector.transform(input_sententce.values)

# 6. 训练逻辑回归模型
# 创建逻辑回归分类器，设置参数：
# max_iter=1000: 最大迭代次数
# C=1.0: 正则化强度（较小的值表示更强的正则化）
# random_state=42: 随机种子，确保结果可重现
model_logistic = linear_model.LogisticRegression(
    max_iter=1000,
    C=1.0,
    random_state=42
)
# 使用特征矩阵和标签（第二列）训练逻辑回归模型
model_logistic.fit(input_feature, dataset[1].values)

# 7. 训练决策树模型
# 创建决策树分类器，设置参数：
# max_depth=10: 限制树的最大深度，防止过拟合
# min_samples_split=5: 分裂节点所需的最小样本数
# random_state=42: 随机种子
model_tree = tree.DecisionTreeClassifier(
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
# 使用特征矩阵和标签训练决策树模型
model_tree.fit(input_feature, dataset[1].values)

# 8. 训练KNN模型
# 创建K近邻分类器，设置参数：
# n_neighbors=5: 考虑最近邻的5个样本
# weights='distance': 根据距离加权，距离越近权重越大
# metric='cosine': 使用余弦距离度量，适合文本数据
model_knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='distance',
    metric='cosine'
)
# 使用特征矩阵和标签训练KNN模型
model_knn.fit(input_feature, dataset[1].values)

# 9. 测试新文本
# 定义测试文本
test_query = "下雨天不出门"
# 对测试文本进行分词处理
test_sentence = " ".join(jieba.lcut(test_query))
# 将测试文本转换为特征向量
test_feature = vector.transform([test_sentence])

# 10. 打印预测结果
print("待预测的文本", test_query)
# 使用KNN模型进行预测
print("KNN模型预测结果: ", model_knn.predict(test_feature))   #输出结果  KNN模型预测结果:  ['Weather-Query']
# 使用逻辑回归模型进行预测
print("LogisticRegression模型预测结果: ", model_logistic.predict(test_feature))  #输出结果  LogisticRegression模型预测结果:  ['Weather-Query']
# 使用决策树模型进行预测
print("决策树模型预测结果: ", model_tree.predict(test_feature))  #输出结果  决策树模型预测结果:  ['Weather-Query']

# 11. 调用大语言模型进行预测
# 创建OpenAI客户端，连接到阿里云的千问模型
client = OpenAI(
    api_key="sk-xxxxxxxxx",  # 这里写自己的api key，作业里就不写出来了，防止盗用
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # API端点
)

# 12. 构建大语言模型的请求
completion = client.chat.completions.create(
    model="qwen-flash",  # 指定使用的模型（千问快速版）
    messages=[
        {
            "role": "user",  # 用户角色
            "content": f"""帮我进行文本分类：{test_query}

输出的类别只能从如下中进行选择，除了类别之外下列的类别，请给出最合适的类别。
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
"""  # 用户输入的内容，包含测试文本和类别约束
        }
    ]
)

# 13. 打印大语言模型的预测结果
print("大语言模型预测结果: ", completion.choices[0].message.content)   #输出结果 大语言模型预测结果:  Weather-Query
