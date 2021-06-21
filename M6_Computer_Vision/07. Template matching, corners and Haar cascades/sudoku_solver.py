import cv2
import sys
import numpy as np
import matplotlib.pyplot as plt

GREEN = (0, 255, 0)
RED = (0, 0, 255)

def fake_sudoku_solver(sudoku: np.ndarray):
    assert sudoku.shape == (9, 9)
    fake_sudoku = sudoku.copy()

    for i in range(fake_sudoku.shape[0]):
        for j in range(fake_sudoku.shape[1]):
            if fake_sudoku[i, j] == 0:
                fake_sudoku[i, j] = np.random.randint(low=1, high=9)
    return fake_sudoku

img = cv2.imread(sys.argv[1])

resized = cv2.resize(img, (364, 364), interpolation=cv2.INTER_LANCZOS4)

sudoku_grey = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
sudoku = resized.copy()

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(sudoku_grey, cv2.COLOR_BGR2RGB))
plt.title('sudoku_grey')
plt.show()

ret, sudoku_th = cv2.threshold(sudoku_grey, 180, 255, cv2.THRESH_BINARY_INV)

# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(sudoku_th, cv2.COLOR_BGR2RGB))
# plt.title('sudoku_th')
# plt.show()


# Finding the lines
lines = cv2.HoughLines(sudoku_th, 1, np.pi/180, sudoku_th.shape[0]-1)

for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(sudoku, (x1, y1), (x2, y2), GREEN, 2)

# plt.figure(figsize=(10,10))
# plt.imshow(cv2.cvtColor(sudoku, cv2.COLOR_BGR2RGB))
# plt.title('sudoku_lines')
# plt.show()

# finding the cells
hsv_lines = cv2.cvtColor(sudoku, cv2.COLOR_BGR2HSV)
H, S, V = cv2.split(hsv_lines)

lower = (40, 0, 0)
upper = (60, 255, 255)

lines_mask = cv2.inRange(hsv_lines, lower, upper)

dilate_kernel = np.ones((5, 5))
lines_mask = cv2.dilate(lines_mask, dilate_kernel, iterations=1)

# plt.figure(figsize=(10,10))
# plt.imshow(lines_mask, cmap='gray')
# plt.title('line_mask')
# plt.show()


cell_contours, h = cv2.findContours(lines_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
# print(len(cell_contours))

cells = resized.copy()
ret, cells_gray = cv2.threshold(cells, 128, 255, cv2.THRESH_BINARY_INV)

# plt.figure(figsize=(10,10))
# plt.imshow(cells_gray, cmap='gray')
# plt.title('cells_gray')
# plt.show()

values = []

# for i, cont in enumerate(cell_contours[1:]):
for i in range(len(cell_contours)-1, 0, -1):
    if cv2.contourArea(cell_contours[i]) > 350:
        x,y,w,h = cv2.boundingRect(cell_contours[i])

        cell = cells_gray[y+2:y+h-2, x+2:x+w-2]

        if cell.sum() == 0:
            values.append(0)
            # cv2.imshow('Blank cell', cell)
            # name = input("Input a name: ")
            # cv2.imwrite(f'./digits/{name}_{i}.jpg', cell)
            # cv2.waitKey(5)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
        else:
            values.append(np.random.randint(low=1, high=9))
            # cv2.imshow('Number cell', cell)
            # name = input("Input a name: ")
            # cv2.imwrite(f'./digits/{name}_{i}.jpg', cell)
            # cv2.waitKey(5)
            # cv2.destroyAllWindows()
            # cv2.waitKey(1)
            

        # cv2.imshow('Cell', cell)
        # cv2.waitKey(200)
        # cv2.destroyAllWindows()
        # cv2.waitKey(1)

        
        cv2.rectangle(cells, (x, y), (x+w, y+h), GREEN, 1)

        # cv2.putText(annoted, text, (x_text_start, y_text_start), text_font, text_scale, text_color, text_thickness)
        # cv2.putText(cells, str(i), (x+w//3, y+h), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 1)

# plt.figure(figsize=(10,10))
# plt.imshow(lines_mask, cmap='gray')
# plt.show()

sudoku_array = np.array(values).reshape((9, 9))
solved_sudoku = fake_sudoku_solver(sudoku_array)

print(sudoku_array)


plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(cells, cv2.COLOR_BGR2RGB))
plt.title('cells')
plt.show()

print(solved_sudoku)

flatten_sudoku = sudoku_array.flatten()
flatten_solved = solved_sudoku.flatten()

# print(flatten_sudoku)
# print(flatten_solved)


# writing the solutions inside the grid
j = 0
for i in range(len(cell_contours)-1, 0, -1):
    if cv2.contourArea(cell_contours[i]) > 350:
        if flatten_sudoku[j] == 0:
            x,y,w,h = cv2.boundingRect(cell_contours[i])
            cv2.putText(resized, str(flatten_solved[j]), (x+10, y+h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 1)
        j += 1


# Showing the solution
plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title('Fake Solved')
plt.show()