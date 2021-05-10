# Alessio

import numpy as np
import time
# setting the random seed
np.random.seed(0)

def merge(A, p, q, r):
    L = [A[p + n] for n in range(q - p + 1)] 
    R = [A[q + m + 1] for m in range(r - q)]
    L.append(float('inf'))
    R.append(float('inf'))
    i = 0
    j = 0
    for k in range(p, r + 1):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1

def merge_sort(A, p, r):
    if p < r:
        q = (p + r) // 2
        merge_sort(A, p, q)
        merge_sort(A, q + 1, r)
        merge(A, p, q, r)

# test the algorithm
simple_list = [5, 7, 8, 4, 9, 2, 6, 3, 0, 1, 19, 11, 15, 18, 10, 13, 16, 14, 12, 17, 20]
merge_sort(simple_list, 0, len(simple_list) - 1)
print(simple_list)


# timing the algorithm
n = 10000  # how many random numbers to be generated
n_times = 1 # how many times perform the sorting, in order to average the timing
times = np.array([])
for i in range(n_times):
    big_list = np.random.permutation(n)
    start = time.time()
    merge_sort(big_list, 0, len(big_list) - 1)
    end = time.time()
    times = np.append(times, end - start)

print(f'merge_sort took {times.mean()} s on average out of {n_times} to sort {n} numbers')