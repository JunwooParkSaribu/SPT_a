import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import numpy as np
import random
import matplotlib.pyplot as plt
from andi_datasets.models_phenom import models_phenom
from andi_datasets.datasets_phenom import datasets_phenom

N = 10000
T = 10000
alpha = 1.99
trajs, labels = models_phenom().single_state(N=N,
                                             T=T,
                                             Ds=[1, 0],
                                             alphas=alpha,
                                             dim=1)


fgn_ = trajs[1:, :, :] - trajs[:-1, :, :]


data_ = []
for n in range(N):
    single_fgn = fgn_[:,n,0]
    for t in range(len(single_fgn) - 1):
        prev = single_fgn[t]
        next = single_fgn[t+1]
        data_.append(next / prev)
        #if (abs(next / prev) > 1000):
        #    print(prev, next, next / prev)
        #    exit()
data_ = np.array(data_)

q1, q2 = np.quantile(data_, [0.025, 0.975])
data_ = data_[(data_ > q1) & (data_ < q2)]

print(np.mean(data_), np.std(data_), data_.shape, np.min(data_), np.max(data_))
plt.figure()
plt.hist(data_, bins=np.linspace(-30, 30, 300))
plt.show()
exit()
    
plt.figure()
plt.hist(np.mean(fgn_, axis=(0, 2)))

plt.figure()
plt.hist(np.var(fgn_, axis=(0, 2)))

plt.figure()
plt.hist(fgn_[:, 0, 0])

plt.show()