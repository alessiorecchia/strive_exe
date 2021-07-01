import os

from uuid import uuid4

import pandas as pd


id = '659a27db-e302-4951-8c17-d838db57be36'
db_path = "./util/db.csv"
a_file = open(db_path, "r")
lines = a_file.readlines()
a_file.close()
# delete lines

for i, line in enumerate(lines):

    print(line.split(',')[0] == id)
    if line.split(',')[0] == id:
        lines.pop(i)

for line in lines:
    print(line)


