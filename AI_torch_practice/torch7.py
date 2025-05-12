import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        # 定义全连接层
        self.fc1 = nn.Linear(input_size, hidden_size)  # 隐藏层
        self.fc2 = nn.Linear(hidden_size, output_size)  # 输出层

    def forward(self, x):
        # 前向传播
        x = torch.relu(self.fc1(x))  # 使用 ReLU 激活函数
        x = self.fc2(x)
        return x

# 定义输入、隐藏和输出层的神经元数量
input_size = 10
hidden_size = 5
output_size = 2

# 实例化模型
model = SimpleNN(input_size, hidden_size, output_size)

# 创建一个随机输入数据（例如，批大小为 1）
input_data = torch.randn(1, input_size)
#print(torch.randn(1, input_size))

# 进行前向传播
output_data = model(input_data)

print("Output:", output_data)