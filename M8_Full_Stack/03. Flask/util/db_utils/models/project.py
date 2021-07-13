from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String
from util.db_utils.connect import db_engine

engine = db_engine()

Base = declarative_base()


class Project (Base):
    __tablename__ = 'projects'
    id = Column(Integer,primary_key=True)
    title = Column(String)
    description = Column(String)
    cover = Column(String)
    live_link = Column(String)
    github_link = Column(String)
    def toJSON(self):
        return {
            'id': self.id,
            'title':self.title,
            'description':self.description,
            'cover':self.cover,
            'live_link':self.live_link,
            'github_link':self.github_link,
        }


Base.metadata.create_all(engine)