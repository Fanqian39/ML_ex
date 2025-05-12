import torch

#创建张量
list_tensor=torch.tensor([[1,2],[3,4]])
print(list_tensor)
zero_tensor=torch.zeros((2,3))
#也可以用step
arrange_tensor=torch.arange(0,5)
linspace_tensor = torch.linspace(0, 1, steps=5)
print(zero_tensor,arrange_tensor,linspace_tensor)

#随机生成张量
#创建一个随机张量
random_tensor=torch.rand((2,3))
#创建一个符合正态分布的张量
normal_tensor=torch.randn((2,3))
print(random_tensor,"\n",normal_tensor)
