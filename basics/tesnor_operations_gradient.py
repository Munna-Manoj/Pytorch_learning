
import torch
#create tensors

x = torch.tensor(3.)
w = torch.tensor(4., requires_grad = True)
b = torch.tensor(5., requires_grad = True)

"""
here we have created three tensors.
x,w,b
As you have already noticed, w and b have additional parameter.
requires_grad = True, 
what does it do???
"""

y = w*x + b 
print(y)

"""
Now, we can compute the derivative of y w.r.t tensors that hae requires_grad set to True i.e w and b.
This is done by y.backward
"""
val = type(y.backward())
#notice that val is of none type, so y.backward() just compute the derivative and those values get stored to respective tensors.
print(val)

#The derivative of y w.r.t the tensors are stored in the .grad property of the respective tensors.
#Display gradients
print("dy/dx:", x.grad)
print("dy/dw:", w.grad)
print("dy/db:", b.grad)

"""
As expected, dy/dw is x=3.
dy/db is 1.
Note that x.grad is None, because x doesn't have requires_grad set to True.
The grad in w.grad stands for gradient, which is the another term for derivative, used mainly when dealing with matrices.
"""

a1 = torch.tensor([[1,2],[3,4.]], requires_grad = True)
x1 = torch.tensor([[2,3,1],[4.,4,7]], requires_grad= True)
b1 = torch.tensor([[5,3,3],[6.,7,9]], requires_grad = True)
y1 = torch.matmul(a1,x1) + b1
print(y1.shape)
y1.backward(torch.ones_like(y1))
#note this point, that backward needs the ones_matrix of the size of output for gradient calculation
print("dy1/dx1:", x1.grad)
print("dy1/da1:", a1.grad)
print("dy1/db1:", b1.grad)

print(a1.shape, x1.shape, b1.shape)


