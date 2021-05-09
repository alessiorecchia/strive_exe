# Alessio

import numpy as np
import time

def merge(A, p, q, r):
    n1 = q - p + 1
    n2 = r - q
    L = [A[p + n] for n in range(n1)] 
    R = [A[q + m + 1] for m in range(n2)]
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
n = 1000000
simple_list = [5, 7, 8, 4, 9, 2, 6, 3, 0, 1, 19, 11, 15, 18, 10, 13, 16, 14, 12, 17, 20]
big_list = np.random.permutation(n)

merge_sort(simple_list, 0, len(simple_list) - 1)
print(simple_list)


# time the algorithm
start = time.time()
merge_sort(big_list, 0, len(big_list) - 1)
end = time.time()

print(f'merge_sort took {end - start} s to sort {n} numbers')