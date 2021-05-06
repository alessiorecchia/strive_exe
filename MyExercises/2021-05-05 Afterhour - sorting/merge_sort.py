import numpy as np

def merge(l, p, q, r):
    n1 = q - p + 1
    n2 = r - q
    L = [float('-inf')] * (n1)
    R = [float('-inf')] * (n2)
    for i in range(n1):
        L[i] = l[p + i]
        for j in range(n2):
            R[j] = l[q + j]
            # L[n1] = float('inf')
            # R[n2] = float('inf')
            L.append(float('inf'))
            R.append(float('inf'))
            i = 0
            j = 0
            for k in range(p, r):
                if L[i] <= R[j]:
                    l[k] = L[i]
                    i += 1
                else:
                    l[k] = R[j]
                    j += 1

            


def merge_sort(l, p, r):
    if p < r:
        q = (p + r) // 2
        merge_sort(l, p, q)
        merge_sort(l, q + 1, r)
        merge(l, p, q, r)

simple_list = [5, 7, 8, 4, 9, 2, 6, 3, 0, 1]

merge_sort(simple_list, 0, 9)

print(simple_list)