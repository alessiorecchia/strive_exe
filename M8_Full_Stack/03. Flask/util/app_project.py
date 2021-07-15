from util.db_utils.connect import db_engine
from util.db_utils.models.project import Project
from sqlalchemy.orm import Session

# db instance
engine = db_engine()

# session instance
session = Session(engine,future=True)


def create_project(title,description,cover, github_link, live_link):
    project = Project(title = title, description = description, cover = cover, github_link = github_link, live_link = live_link)
    session.add(project)
    session.commit()
    return project.toJSON()


def get_projects():
    result = [row.toJSON() for row in session.query(Project)]
    return result

def get_projects_by_id(id):
    project = session.query(Project).get(id)
    # print('\n\n\n', project.toJSON(), '\n\n\n')
    if(project):
        return project.toJSON()
    return None
    


def update_project_by_id(id,update):
    print('\n\n\n', id, '\n\n\n')
    project = session.query(Project).get(id)
    if(project):
        for key,value in update.items():
            setattr(project,key,value)
        session.commit()
        return project.toJSON()
    return None
    


def delete_project_by_id(id):
    project = session.query(Project).get(id)
    if(project):
        session.delete(project)
        session.commit()
        return project.toJSON()
    return None
    





# create_project('Merda','merda','merda','merda', 'merda')
# update_project_by_id(1, {'title': 'Merdazza', 'cover': 'https://www.merdazzadek.it/mrddk.png'})
# prj = get_projects_by_id(1)
# print(prj)
# delete_project_by_id(1)
