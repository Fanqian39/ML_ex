import torch

# 定义张量并启用梯度计算
x = torch.tensor(2.0, requires_grad=True)

# 定义一个函数
y = x**3 + 5*x**2 + 10

# 执行反向传播
#执行后.grad属性将直接包含该标量函数相对于该张量的梯度（因为前面设置了 requires_grad=True）
c=y.backward()

# 查看梯度
print(f"f(x) = {y.item()} at x = {x.item()}")
print(f"f'(x) = {x.grad.item()}")