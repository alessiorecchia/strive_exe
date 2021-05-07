import numpy as np

def merge(A, p, q, r):
    n1 = q - p + 1
    n2 = r - q
    L = [A[n] for n in range(0, n1)] 
    R = [A[m] for m in range(n2, r + 1)]
    L.append(float('inf'))
    R.append(float('inf'))
    i = 0
    j = 0
    for k in range(p, r):
        if L[i] <= R[j]:
            A[k] = L[i]
            i += 1
        else:
            A[k] = R[j]
            j += 1

    return A

            


def merge_sort(A, p, r):
    if p < r:
        q = (p + r) // 2
        merge_sort(A, p, q)
        merge_sort(A, q + 1, r)
        merge(A, p, q, r)

    return A

simple_list = [5, 7, 8, 4, 9, 2, 6, 3, 0, 1]

merge_sort(simple_list, 0, len(simple_list))

print(simple_list)