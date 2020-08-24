from sqlalchemy import Column,Integer, String,DECIMAL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool
Base = declarative_base()


class Followers1(Base):
    __tablename__ = 'followers'
    count = Column(Integer, primary_key=True, autoincrement=True)
    mainuser = Column(String(250), nullable=False)
    id = Column(Integer)
    name = Column(String(250), nullable=False)
    screen_name = Column(String(250), nullable=False)
    profile_image_url = Column(String(250))
    tweet = Column(String(250))
    lg_result = Column(String(250))
    finalresult = Column(DECIMAL(4, 2))



class Result(Base):
    __tablename__ = 'result'
    mainuser = Column(String(250), nullable=False)
    fid = Column(Integer, primary_key=True)
    fname = Column(String(250), nullable=False)
    fscreen_name = Column(String(250), nullable=False)
    fprofile_image_url = Column(String(250))
    finalresult = Column(DECIMAL(4, 2))


engine = create_engine('sqlite:///followersdatabase.db',connect_args={'check_same_thread':False},poolclass=StaticPool)

Base.metadata.create_all(engine)
