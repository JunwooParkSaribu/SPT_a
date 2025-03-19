import os
import sys
sys.path.insert(0, os.path.abspath('../'))
import numpy as np
import random
import matplotlib.pyplot as plt
from andi_datasets.models_phenom import models_phenom
from andi_datasets.datasets_phenom import datasets_phenom

def cor(time_lag, hurst_index):
    sigma = 1
    cor_ = abs(time_lag)**(2*hurst_index) * (sigma**2/2) * (abs(1 - 1/time_lag)**(2*hurst_index) + abs(1 + 1/time_lag)**(2*hurst_index) - 2)
    return cor_

x_ = np.linspace(-10, 10, 1000)
y_ = cor(x_, 0.01)

plt.figure()
plt.plot(x_, y_)
plt.show()
exit()

N = 10000
T = 10000
alpha = 0.01
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

plt.figure()
plt.hist(fgn_.flatten(), bins=np.linspace(-30, 30, 300))
a = fgn_.flatten().copy()
np.random.shuffle(a)

plt.figure()
plt.hist(fgn_.flatten() / a, bins=np.linspace(-30, 30, 300))
"""
gauss1 = np.random.normal(0, 1.4, size=10000)
gauss2 = np.random.normal(0, 1.4, size=10000)
x_path = np.linspace(-30, 30, 300)
def func1(x):
    return np.exp(-1/2 * ())
plt.figure()
plt.hist(gauss1 / gauss2, bins=np.linspace(-30, 30, 300))
plt.show()

print(np.mean(fgn_.flatten()), np.std(fgn_.flatten()))
plt.figure()
plt.hist(fgn_.flatten(), bins=np.linspace(-30, 30, 300))
"""
print(np.mean(data_), np.std(data_), data_.shape, np.min(data_), np.max(data_))

plt.figure()
plt.hist(data_, bins=np.linspace(-30, 30, 300))

plt.figure()
plt.hist(np.random.normal(np.mean(data_), np.std(data_), size=10000), bins=np.linspace(-30, 30, 300))

plt.show()
exit()
    
plt.figure()
plt.hist(np.mean(fgn_, axis=(0, 2)))

plt.figure()
plt.hist(np.var(fgn_, axis=(0, 2)))

plt.figure()
plt.hist(fgn_[:, 0, 0])

plt.show()