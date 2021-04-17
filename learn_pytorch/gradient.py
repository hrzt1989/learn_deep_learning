import torch
# a = torch.ones(3, requires_grad=True)
# print(a)
# b = a + 1
# print(b)
# print(b.grad_fn)
# print(a.is_leaf, b.is_leaf)
# print(b.requires_grad)
# c = torch.rand(1,3)
# print(c.requires_grad)
# c.requires_grad_(True)
# print(c.requires_grad)

# a = torch.ones(1, 3, requires_grad=True)
# b = a + 1
# print('b.requires_grad', b.requires_grad)
# z = a * b
# out = z.mean()
# print('out',out)
# out.backward(retain_graph=True)
# print('a.grad', a.grad)
# print('b.grad', b.grad)
# print(a.requires_grad)
# print(a)
# out2 = z.mean()
# print('out2',out2)
# out2.backward()
# print('a.grad2', a.grad)

# x = torch.ones(2, 2, requires_grad=True)
# out2 = x.sum()
# out2.backward()
# print(x.grad)
#
# out3 = x.sum()
# print('out3', out3)
# # x.grad.data.zero_()
# out3.backward()
# print(x.grad)

# 1/3(a1^2 + a1 + a2^2 + a2 + a3^2 + a3)
#
# 1/3(2 * a1 + 1)

# a = torch.ones(1, 5, requires_grad=True)
# print(a.data)
# print(a.data.requires_grad)
# b = a + 1
# z = a * b
# a.data *= 100
# print(z)
# out = z.mean()
# out.backward()
# print(a)
# print(a.grad)
a = torch.ones(2,2)
b = torch.ones(2, 1)
c = torch.mm(a, b)
print(a, b, c)

x = torch.ones(1, 5, requires_grad=True)

print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

x = torch.ones(1,requires_grad=True)

print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

y = x + 1
z = y * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

z.backward()
print(x) # 更改data的值也会影响tensor的值
print(x.grad)