import numpy as np
from time import time

big_list = np.random.permutation(1000000)

simple_list = np.arange(20)

def copy_for(A):
    new_A = np.zeros(A.shape)
    for i in range(len(A)):
        new_A[i] = A[i] + 1
    return new_A

def copy_lc(A):
    new_A = [(A[i] + 1) for i in range(A.shape[0])]
    return new_A

new = copy_for(simple_list)
print(new)

new_lc = copy_lc(simple_list)
print(new_lc)

start = time()
copy_for(big_list)
end = time()
print(end-start)

start2 = time()
copy_lc(big_list)
end2 = time()
print(end2-start2)