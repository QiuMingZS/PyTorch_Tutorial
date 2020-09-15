# -*- coding: UTF-8 -*-
# No.1
import torch 
from time import time
a = torch.rand(10000, 10000)
b = torch.rand(10000, 10000)
start1 = time()
print(a.matmul(b))
print('Total Time1: ', time() - start1)

a = a.cuda()
b = b.cuda()
start2 = time()
print(a.matmul(b))
print('Total Time2: ', time() - start2)


