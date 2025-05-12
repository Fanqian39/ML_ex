import torch
import torch.nn as nn

# 定义Sigmoid激活函数
#适合用于二分类问题的输出层
sigmoid = nn.Sigmoid()

# 测试输入
input_tensor = torch.tensor([-1.0, 0.0, 1.0])
output_tensor = sigmoid(input_tensor)

print(output_tensor)  # 输出各个值的Sigmoid结果

# 定义Tanh激活函数
#在零附近有更强的非线性，通常能带来更好的收敛效果
tanh = nn.Tanh()
output_tensor = tanh(input_tensor)

print(output_tensor)  # 输出各个值的Tanh结果

# 定义ReLU激活函数
#能有效缓解梯度消失问题
relu = nn.ReLU()
output_tensor = relu(input_tensor)

print(output_tensor)  # 输出ReLU结果

# 定义Leaky ReLU激活函数
#解决了ReLU的“神经元死亡”问题
leaky_relu = nn.LeakyReLU(negative_slope=0.01)
output_tensor = leaky_relu(input_tensor)

print(output_tensor)  # 输出Leaky ReLU结果

# 定义Softmax激活函数
#将模型的输出转换为概率分布
softmax = nn.Softmax(dim=0)
output_tensor = softmax(input_tensor)

print(output_tensor)  # 输出Softmax结果