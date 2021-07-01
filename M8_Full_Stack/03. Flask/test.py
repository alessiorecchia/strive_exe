import os

from uuid import uuid4

import pandas as pd


id = 'e6fc0220-9943-4a84-8b88-b9a0765e813f'
db_path = "./util/db.csv"
a_file = open(db_path, "r")
lines = a_file.readlines()
a_file.close()
# delete lines

for i, line in enumerate(lines):

    print(line.split(',')[0] == id)
    if line.split(',')[0] == id:
        lines = lines.pop(i)
print(len(lines))

for line in lines:
    print(lines)

new_file = open(db_path, "w+")
for line in lines:
    new_file.write(line)
new_file.close()

