from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import os

# 1. 初始化 FastAPI 应用
app = FastAPI()

# 2. 加载模型 (模拟加载过程)
# 运维文档要求：模型文件需放置在 models/ 目录下
model_path = os.path.join('models', 'intent_model.pkl')
if os.path.exists(model_path):
    print(f"正在加载模型: {model_path}")
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
else:
    print("警告：未找到模型文件，请先运行 train.py")

# 定义输入数据的格式 (接收一个 text 文本)
class Item(BaseModel):
    text: str

# 3. 定义接口 (符合背景文档要求：接收输入，返回意图)
@app.post("/predict")
async def predict_intent(item: Item):
    user_input = item.text
    
    # --- 这里是模拟模型的推理逻辑 ---
    # 背景文档要求识别"导航"、"媒体控制"等意图
    if "导航" in user_input:
        intent = "navigation"
        confidence = 0.98
    elif "空调" in user_input or "温度" in user_input:
        intent = "ac_control"
        confidence = 0.95
    else:
        intent = "unknown"
        confidence = 0.50
    # -----------------------------

    return {
        "text": user_input,
        "intent": intent,
        "confidence": confidence
    }