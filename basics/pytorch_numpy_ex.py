import numpy as np
import torch
x = np.array([[1,2],[3,4.]])
print(x)

#convert numpy.array to torch.tensor by torch.from_numpy
y = torch.from_numpy(x)
print(y)

#note that, when we pass numpy array to tensor, it uses same memory location as numpy
#this means that any changes to x bring same changes to y
x[:,-1] = 2
print(x)
print(y)


#but
t = torch.tensor([[1,2],
            [3,4]])
print(t)
#here, t copies the array to some memroy location

print(x.dtype, y.dtype)

#converting the tensor to numpy array
z = t.numpy()
print(z)
print(type(z))



