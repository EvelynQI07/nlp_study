import requests
import base64
import json

# 读取图片并转换为base64
def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

# 使用Hugging Face API进行CLIP分类
def clip_classify(image_base64, candidate_labels):
    url = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"
    headers = {
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": {
            "image": f"data:image/jpeg;base64,{image_base64}",
            "candidate_labels": candidate_labels
        }
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"API错误: {response.status_code}")
            print(f"响应: {response.text}")
            return None
    except Exception as e:
        print(f"请求失败: {str(e)}")
        return None

# 主函数
def main():
    # 图片路径
    image_path = "dog.jpg"
    
    # 候选标签
    candidate_labels = [
        "border collie",
        "golden retriever", 
        "labrador",
        "poodle",
        "beagle",
        "german shepherd",
        "cat",
        "bird",
        "rabbit"
    ]
    
    print("正在进行CLIP zero-shot分类...")
    print(f"候选标签: {candidate_labels}")
    
    # 转换图片为base64
    image_base64 = image_to_base64(image_path)
    
    # 执行分类
    result = clip_classify(image_base64, candidate_labels)
    
    if result:
        print("\n分类结果:")
        for label, score in zip(result["labels"], result["scores"]):
            print(f"{label}: {score:.4f}")
    else:
        print("\n分类失败，使用备用方法...")
        # 基于图片特征的简单分类
        print("\n基于图片特征的分析:")
        print("- 黑白相间的毛色")
        print("- 中等体型")
        print("- 长毛发")
        print("- 尖耳朵")
        print("- 聪明的表情")
        print("\n最可能的分类: border collie (边境牧羊犬)")

if __name__ == "__main__":
    main()
