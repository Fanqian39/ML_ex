import torch

a = torch.tensor([[1, 2], [3, 4]])
b = torch.tensor([[5, 6], [7, 8]])

#张量乘法
#元素乘法
c=a*b
print(c)
#矩阵乘法
d=torch.mm(a,b)
print(d)

#转置
e=a.t()
print(e)

#连接
#沿着行连接
concat_dim0=torch.cat((a,b),dim=0)
#沿着列连接
concat_dim1=torch.cat((a,b),dim=1)
print(concat_dim0,"\n",concat_dim1)

#数值统计：求和、均值、方差等
sum_a=torch.sum(a)
#必须转换成浮点数
mean_a=torch.mean(a.float())
std_a=torch.std(a.float())
print(sum_a)
print(mean_a)
print(std_a)
