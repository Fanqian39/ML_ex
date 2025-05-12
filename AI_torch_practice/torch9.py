import torch
import torch.nn as nn
import torch.optim as optim


# 创建一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


# 初始化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.MSELoss()  # 策略（定义损失函数）
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 待讲解的优化器

# 示例输入输出
x = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
y = torch.tensor([[2.0], [4.0], [6.0]])

# 训练过程
for epoch in range(100):
    model.train()

    # 清零梯度
    optimizer.zero_grad()

    # 前向传播
    outputs = model(x)

    # 计算损失
    loss = criterion(outputs, y)  # 计算损失
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # 反向传播
    loss.backward()

    # 更新参数
    optimizer.step()