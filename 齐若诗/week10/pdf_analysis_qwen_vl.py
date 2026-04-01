import os
import base64
import requests
import PyPDF2
from PIL import Image, ImageDraw, ImageFont
import io

# 从PDF提取文本
def extract_text_from_pdf(pdf_path):
    """从PDF第一页提取文本"""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            if len(reader.pages) > 0:
                page = reader.pages[0]
                text = page.extract_text()
                return text
            else:
                print("PDF没有页面")
                return None
    except Exception as e:
        print(f"PDF文本提取失败: {str(e)}")
        return None

# 文本转图像
def text_to_image(text, width=800, height=1000):
    """将文本转换为图像"""
    try:
        # 创建空白图像
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # 设置字体（使用默认字体）
        try:
            font = ImageFont.truetype('/System/Library/Fonts/STHeiti Light.ttc', 12)
        except:
            font = ImageFont.load_default()
        
        # 绘制文本
        lines = text.split('\n')
        y_position = 50
        line_height = 20
        
        for line in lines:
            if y_position > height - 50:
                break
            draw.text((50, y_position), line, fill='black', font=font)
            y_position += line_height
        
        return image
    except Exception as e:
        print(f"文本转图像失败: {str(e)}")
        return None

# 图像转base64
def image_to_base64(image):
    """将图像转换为base64编码"""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# 调用Qwen-VL模型
def analyze_image_with_qwen_vl(image_base64, prompt, api_key):
    """使用Qwen-VL模型分析图像"""
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    payload = {
        "model": "qwen3-vl-plus",  # 可根据需要更换模型
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "stream": False
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
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
    # 配置
    pdf_path = "example.pdf"  # 替换为实际PDF路径
    api_key = os.getenv("DASHSCOPE_API_KEY")  # 从环境变量获取API Key
    
    if not api_key:
        print("请设置DASHSCOPE_API_KEY环境变量")
        print("示例: export DASHSCOPE_API_KEY=your_api_key")
        return
    
    if not os.path.exists(pdf_path):
        print(f"PDF文件不存在: {pdf_path}")
        return
    
    # 1. 从PDF提取文本
    print("正在提取PDF文本...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        return
    
    print("提取的文本:")
    print(text)
    print("\n")
    
    # 2. 将文本转换为图像
    print("正在创建文本图像...")
    image = text_to_image(text)
    if not image:
        return
    
    # 3. 将图像转换为base64
    print("正在编码图像...")
    image_base64 = image_to_base64(image)
    
    # 4. 定义分析提示
    prompt = "请详细分析这张图片中的文本内容，包括结构、格式和主要信息"
    
    # 5. 调用Qwen-VL模型
    print("正在分析内容...")
    result = analyze_image_with_qwen_vl(image_base64, prompt, api_key)
    
    # 6. 处理结果
    if result:
        print("\n分析结果:")
        print(result['choices'][0]['message']['content'])
    else:
        print("分析失败")

if __name__ == "__main__":
    main()
