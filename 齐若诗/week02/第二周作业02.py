#第二项作业 调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

# 1. 生成模拟数据：构建 sin 函数
# 生成 -2π 到 2π 之间的 200 个点
X_numpy = np.linspace(-2 * np.pi, 2 * np.pi, 200).reshape(-1, 1)
# y = sin(x) 并添加少量噪声
y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(200, 1)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("Sin 函数数据生成完成。")
print("---" * 10)

# 2. 定义多层神经网络类
# 线性模型无法拟合曲线，因此需要增加隐藏层和激活函数
class SinModel(nn.Module):
    def __init__(self):
        super(SinModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 64),  # 输入层到隐藏层1
            nn.ReLU(),  # 激活函数提供非线性能力
            nn.Linear(64, 64),  # 隐藏层2
            nn.ReLU(),
            nn.Linear(64, 1)  # 输出层
        )

    def forward(self, x):
        return self.net(x)
model = SinModel()

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()  # 回归任务使用均方误差
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam 通常比 SGD 更快收敛

# 4. 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()

    # 前向传播
    y_pred = model(X)
    # 计算损失
    loss = loss_fn(y_pred, y)
    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数
    # 每 100 个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 绘制结果
model.eval()
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.5)
plt.plot(X_numpy, y_predicted, label='Multi-layer Fit', color='red', linewidth=3)
plt.legend()
plt.grid(True)
plt.title("Multi-layer Neural Network Fitting Sin Function")

# 保存图片以避免 FigureCanvasInterAgg 报错
plt.savefig('sin_fit_result.png')
print("\n绘图完成！结果已保存为 'sin_fit_result.png'")
