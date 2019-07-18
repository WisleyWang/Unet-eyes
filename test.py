import os
import numpy as np
# for i,j,k in os.walk('./DRIVE/'):
#     # print(i)
#     # print(j)
#     print(k)

# print((os.listdir(r'.\DRIVE\training\images\\')))

#
# a=np.asarray([[1,2]])
# print(a.shape)
# b=np.zeros((2,2))
# print(b.shape)
# c=np.concatenate((b,a),axis=0)
# print(c)
# x2=np.arange(12).reshape(3,4)
# print(x2)
# print(np.linalg.norm(x2,ord=2))
# print(sorted([2,3,1]))
a=np.random.randint(0,10,16).reshape((4,4))
print(a)
image_mean = np.mean(a)
image_std = np.std(a)
imgs_normalized = (a - image_mean) / image_std
print(imgs_normalized[2,0])
print(imgs_normalized)