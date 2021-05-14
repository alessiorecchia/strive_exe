vampire = []
for i in range(1, 9):
    for j in range(0, 9):
        for k in range(1, 9):
            for l in range(1, 9):
                n = (i * 10 + j) * (k * 10 + l)
                if (n == i * 10**3 + j * 10**2 + k * 10 + l) or (n == i * 10**3 + k * 10**2 + j * 10 + l) or (n == i * 10**3 + k * 10**2 + l * 10 + j):
                    vampire.append([i*10 + j, k*10 + l, n])

print(vampire)