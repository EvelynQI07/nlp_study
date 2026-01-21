# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import text_classify_ml as ml
import text_classify_llm as llm
from fastapi import FastAPI
import uvicorn
app = FastAPI()

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

@app.get("/text-cls/ml")
def text_classify_ml(text:str) -> str:
    return ml.text_classify(text)

@app.get("/text-cls/llm")
def text_classify_llm(text:str) -> str:
    return llm.text_classify(text)


#print(ml.text_classify("帮帮我导航到春熙路"))
# print(llm.text_classify("帮帮我导航到春熙路"))

# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    #命令行启动 uvicorn main:app --port 8888 --reload
    #uvicorn.run(app, host="127.0.0.1", port=8888, reload=True)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
