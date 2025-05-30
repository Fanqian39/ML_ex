import torch
import torch.nn as nn

class SimpleNN(nn.Module):  #所有的神经网络模块都需要继承自nn.Module，它是PyTorch中所有神经网络模块的基类
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(10, 5)  # 输入层到隐藏层 #定义输入及输出的维数
        self.fc2 = nn.Linear(5, 1)  # 隐藏层到输出层
        self.relu = nn.ReLU()  # 使用ReLU激活函数
        self.sigmoid = nn.Sigmoid()  # 使用Sigmoid激活函数

    def forward(self, x):
        x = self.fc1(x)  # 前向传播到隐藏层
        x = self.relu(x)  # 激活
        x = self.fc2(x)  # 前向传播到输出层
        x = self.sigmoid(x)  # 激活
        return x

# 创建一个神经网络实例
model = SimpleNN()
print(model)