# -*- coding: UTF-8 -*-
# No.0
import torch 
import glob
import numpy as np
import matplotlib.pyplot as plt
import pylab
from PIL import Image

# 1d vector with only an element -> 0d
x = torch.rand(1)
print(x)

# 1d vector
x = torch.rand(10)
print(x, x.size())

y = torch.FloatTensor([23, 24, 25, 26, 23, 26.2, 36])
print(y, y.size())

# 2d vector
z = torch.rand(2, 3)
print(z, z.size())

# 3d vector
J20 = np.array(Image.open('./pics/J20.jpg').resize((2000, 1334)))
J20_tensor = torch.from_numpy(J20)
J20_tensor.size()

plt.imshow(J20)
# pylab.show()

plt.imshow(J20_tensor[:,:,0].numpy())
# pylab.show()

plt.imshow(J20[500:700, 300:500, :])
# pylab.show()

# 4d 
fighters = glob.glob('./pics/*.jpg')
fighter_imgs = np.array([np.array(Image.open(fighter).resize((224,224))) for fighter in fighters[:2]])
fighter_imgs = fighter_imgs.reshape(-1, 224, 224, 3)
fighter_tensors = torch.from_numpy(fighter_imgs)
print(fighter_tensors.size())

