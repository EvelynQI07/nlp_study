from pydantic import BaseModel, Field

class TranslationIntent(BaseModel):
    """
    识别并精准提取用户的文本翻译需求。
    需要将用户的操作指令与实际需要翻译的文本分离开来。
    """
    source_language: str = Field(
        default="自动识别", 
        description="原始语种。如果用户话语中没有明确指明原来的语言是什么，请固定输出 '自动识别'。"
    )
    target_language: str = Field(
        description="目标语种。例如 '中文', '英文', '日文' 等。如果用户未指明，请根据常理推断或输出 '未知'。"
    )
    text_to_translate: str = Field(
        description="需要被翻译的纯文本内容。必须剔除掉类似 '帮我翻译'、'请把这句翻译成' 等指令性废话，只保留核心待翻译实体。"
    )

# 实例化智能体
translator_agent = ExtractionAgent(model_name="qwen-plus")

# --- 测试用例集 ---
test_prompts = [
    "帮我将 good! 翻译为中文",  # 基础测试：缺失原始语种，带有指令废话
    "请把日语的 'こんにちは' 翻译成法语",  # 明确双语种测试
    "我马上要开跨国会议了，快帮我把这句改成英文：'我们的服务器出故障了，预计两小时后恢复。'", # 复杂业务场景，包含背景噪音
]

print("="*40)
print("开始测试翻译智能体信息抽取")
print("="*40)

for prompt in test_prompts:
    print(f"\n[用户输入]: {prompt}")
    result = translator_agent.call(prompt, TranslationIntent)
    
    if result:
        print(f" └─ [原始语种]: {result.source_language}")
        print(f" └─ [目标语种]: {result.target_language}")
        print(f" └─ [待译文本]: {result.text_to_translate}")
    else:
        print(" └─ [抽取失败]")


# 运行测试代码
if __name__ == '__main__':
    # 实例化你的智能体
    translator_agent = ExtractionAgent(model_name="qwen-plus")
    
    # 你指定的测试用例
    test_prompt = "帮我将 good! 翻译为中文"
    
    print("="*40)
    print(f"[输入文本]: {test_prompt}")
    print("="*40)
    
    # 调用智能体进行信息抽取
    result = translator_agent.call(test_prompt, TranslationIntent)
    
    if result:
        print(f"【原始语种】: {result.source_language}")
        print(f"【目标语种】: {result.target_language}")
        print(f"【待翻译的文本】: {result.text_to_translate}")
    else:
        print("抽取失败，请检查网络或大模型调用状态。")

#下面是结果输出：
========================================
[输入文本]: 帮我将 good! 翻译为中文
========================================
【原始语种】: 自动识别 (或 英文)
【目标语种】: 中文
【待翻译的文本】: good!


