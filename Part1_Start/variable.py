# -*- coding: UTF-8 -*-
# No.2
import torch 
from torch.autograd import Variable
x = Variable(torch.ones(2, 2), requires_grad=True)
y = x.mean()

y.backward()

print(x.grad)
print(x.grad_fn)
print(y.grad_fn)