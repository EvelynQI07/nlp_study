from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# 创建测试PDF
def create_test_pdf():
    c = canvas.Canvas('example.pdf', pagesize=A4)
    width, height = A4
    
    # 添加标题
    c.setFont('Helvetica-Bold', 18)
    c.drawString(100, height - 100, '测试文档')
    
    # 添加正文
    c.setFont('Helvetica', 12)
    text = '''这是一个测试PDF文档，用于测试Qwen-VL模型的PDF解析功能。

文档包含以下内容：
1. 标题和正文文本
2. 一个简单的表格
3. 一些列表项

表格示例：
| 姓名 | 年龄 | 职业 |
|------|------|------|
| 张三 | 25   | 工程师 |
| 李四 | 30   | 设计师 |

列表项：
- 项目一
- 项目二
- 项目三

希望Qwen-VL能够准确识别这些内容。'''
    
    # 绘制文本
    text_lines = text.split('\n')
    y_position = height - 150
    for line in text_lines:
        c.drawString(100, y_position, line)
        y_position -= 20
    
    # 保存PDF
    c.save()
    print("测试PDF创建成功: example.pdf")

if __name__ == "__main__":
    create_test_pdf()
