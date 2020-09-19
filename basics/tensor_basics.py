import torch
t1 = torch.tensor(3.)
print(t1)
print(t1.shape)


x = torch.tensor([1,3,4], dtype= float)
print(x)
print(x.shape)
#x.dtype will gives us the data type



#matrix
#dimension of the matrix should be consistent
x1 = torch.tensor([[1,2,3],[3,4,5],[3,4,6]])
print(x1)
print(x1.shape)


#3D matrix
x2 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
print(x2)
print(x2.shape)




