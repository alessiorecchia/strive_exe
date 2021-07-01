import os

from uuid import uuid4


def read_db():
    db_path = "./util/db.csv"
    projects = []
    with open(db_path) as f:
        lines = f.readlines()
        for line in lines:
            if(len(line) != 0):
                array = line.split(",")
                projects.append(array)

    return projects


def write_project(title, description, cover, githubLink, liveLink):
    line = "{},{},{},{},{},{}".format(uuid4(), title, description, cover, githubLink, liveLink)
    db_path = "./util/db.csv"
    with open(db_path, 'r+') as f:
        lines = f.readlines()
        if(len(lines) == 0):
            f.write(f'{line}')
        else:
            f.write(f'\n{line}')
    return None


def find_project_by_id(id):
    projects = read_db()
    found = None
    for project in projects:
        project_id = project[0]
        if(project_id == id):
            return project
    return found


def find_project_by_id_and_delete(id):
    db_path = "./util/db.csv"
    a_file = open(db_path, "r")
    lines = a_file.readlines()
    a_file.close()
    # delete lines
    for i, line in enumerate(lines):
        if line.split(',')[0] == id:
            lines.pop(i)
    # write to file without line
    new_file = open(db_path, "w+")
    for line in lines:
        new_file.write(line)
    new_file.close()

    return None
