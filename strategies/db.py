from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

engine = create_engine('sqlite:///database/database.db')
Session = sessionmaker(bind=engine)
current_session = scoped_session(Session)
Base = declarative_base()


class Strategies(Base):
    __tablename__ = 'strategies'
    id = Column(Integer, primary_key=True)
    Strategy = Column(String(100))
    Status = Column(String(100))
    Time = Column(DateTime)
    Open = Column(Float)
    Close = Column(Float)
    Lag = Column(Integer)
    Signal = Column(Integer)
    Rule = Column(String(100))
    Result = Column(Float)
    Cumsum = Column(Float)


Base.metadata.create_all(engine)
