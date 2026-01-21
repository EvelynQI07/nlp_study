from typing import Union

from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "你好学习机器学习和大模型的同学！"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}

# fastapi run main.py

"""
fastapi http服务的部署框架，开发框架；
本地的程序 部署 为http服务： 别人（别人、用户、其他客户端）请求
"""