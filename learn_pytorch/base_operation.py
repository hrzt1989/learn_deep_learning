import torch
import numpy
if '__main__' == __name__:
    x = torch.rand(5, 3)
    # y = torch.rand(5,3)
    # y = torch.zeros(5,3)
    # y = torch.tensor([5.5, 3])
    #
    # a = torch.tensor([1,2,3])
    # b = torch.tensor([1,5,6])
    # a1 =a.view(3, 1).clone()
    # print(a)
    # print(a1)
    # print(a.size())
    # print(a1.size())
    # print(a1 + 1)

    # y = x.new_ones(5,3, dtype=torch.float64)
    # z = torch.rand_like(x, dtype=torch.float)
    # print(x)
    # print(y)
    # print(z)

    # a =  torch.arange(1,4).view(-1, 3)
    # b = torch.arange(2,5).view(3,1)
    # print(a)
    # print(b)
    # print( a + b)
    # a = torch.ones(1,3)
    # b = a.numpy()
    # print(a)
    # print(b)
    #
    # b += 1
    # print(a)
    # a = torch.rand(3)
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    #     b = torch.ones_like(a, device=device)
    #     print(b)
    #     a1 = a.to(device)
    #     a1 += 1
    #     print(a)
    #     print(a1)
    #     z = a1 + b
    #     print(z)
    #     a = torch.arange(1,7).view(2,3)
    #     print(a * a)

    X = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(X.sum(dim=0,keepdim=True))
    print(X.sum(dim=1, keepdim=True))
    print(X)
