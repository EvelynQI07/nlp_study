import requests
from PIL import Image
import base64
import json
import io

# 保存图片
def save_image_from_base64(base64_str, output_path):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    image.save(output_path)
    return output_path

# 使用Hugging Face API进行CLIP分类
def clip_zero_shot_classification(image_path, candidate_labels):
    # 读取图片并转换为base64
    with open(image_path, "rb") as f:
        image_data = f.read()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # 准备请求数据
    payload = {
        "inputs": {
            "image": f"data:image/jpeg;base64,{base64_image}",
            "candidate_labels": candidate_labels
        }
    }
    
    # 发送请求到Hugging Face的CLIP模型
    try:
        response = requests.post(
            "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32",
            headers={"Authorization": "Bearer hf_your_token_here"},  # 替换为你的Hugging Face token
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result
        else:
            print(f"API请求失败: {response.status_code}")
            print(f"错误信息: {response.text}")
            return None
    except Exception as e:
        print(f"请求过程中出错: {str(e)}")
        return None

# 主函数
def main():
    # 图片路径
    image_path = "dog.jpg"
    
    # 候选标签
    candidate_labels = ["border collie", "golden retriever", "labrador", "poodle", "beagle", "german shepherd", "cat", "bird", "rabbit"]
    
    print("正在进行CLIP zero-shot分类...")
    print(f"候选标签: {candidate_labels}")
    
    # 执行分类
    result = clip_zero_shot_classification(image_path, candidate_labels)
    
    if result:
        print("\n分类结果:")
        for label, score in zip(result["labels"], result["scores"]):
            print(f"{label}: {score:.4f}")
    else:
        print("分类失败，请检查网络连接或API token")

if __name__ == "__main__":
    main()
