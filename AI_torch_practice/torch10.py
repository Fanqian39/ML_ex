import torch
import torch.nn as nn
import torch.optim as optim

# 创建一个简单的线性模型
model = nn.Linear(10, 1)

# 选择损失函数
criterion = nn.MSELoss()

# 使用SGD优化器
optimizer_sgd = optim.SGD(model.parameters(), lr=0.01)

# 或者使用Adam优化器
optimizer_adam = optim.Adam(model.parameters(), lr=0.001)

# 假设有一些输入数据和目标标签
inputs = torch.randn(5, 10)
targets = torch.randn(5, 1)
print(inputs)
print(targets)

# 进行一次训练迭代
optimizer_sgd.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer_sgd.step()
print(outputs)

# 如果使用Adam优化器
optimizer_adam.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer_adam.step()
print(outputs)

print("训练完成")