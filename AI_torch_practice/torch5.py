import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

#设置随机种子
torch.manual_seed(42)

#生成数据
x=torch.linspace(0,1,100).reshape(-1,1) # 100个数据点
y=2*x+1+torch.randn(x.size())*0.1 # y = 2x + 1 + 噪声
#print(x,'\n',y)

#可视化
plt.scatter(x.numpy(),y.numpy(),color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Generated Data')
plt.show()

#定义模型
model=nn.Linear(1,1) # 输入是1维，输出也是1维

#定义损失函数和优化器
criterion=nn.MSELoss()
optimizer=torch.optim.SGD(model.parameters(),lr=0.01)

#训练模型
num_epochs=200

for epoch in range(num_epochs):
    #正向传播 #从输入到输出的计算过程，用于生成预测结果
    outputs=model(x)  #计算预测
    loss=criterion(outputs,y)  #计算损失

    #反向传播 #从输出到输入的梯度计算过程，用于更新模型参数。
    optimizer.zero_grad()  #清零之前的梯度 #默认情况下，梯度是累加的
    loss.backward()  #计算梯度
    optimizer.step()  #更新参数

    if (epoch+1)%20==0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 可视化结果
with torch.no_grad():  # 在这个上下文中不需要计算梯度 #上下文管理器
    predicted = model(x)

plt.scatter(x.numpy(), y.numpy(), color='blue')
plt.plot(x.numpy(), predicted.numpy(), color='red', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend(['Original', 'Predicted'])
plt.show()