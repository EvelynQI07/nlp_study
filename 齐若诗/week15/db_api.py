from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import declarative_base, relationship, sessionmaker
from datetime import datetime

import yaml  # type: ignore

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

db_config = config['database']
db_type = db_config['engine']

if db_type == "sqlite":
    db_path = db_config.get('path', 'rag.db')
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
else:
    host = db_config.get('host', 'localhost')
    port = db_config.get('port', 3306)
    username = db_config.get('username', 'user')
    password = db_config.get('password', 'password')
    database = db_config.get('database', 'mydb')
    engine = create_engine(
        f"{db_type}://{username}:{password}@{host}:{port}/{database}",
        echo=False
    )

Base = declarative_base()


class KnowledgeDatabase(Base):
    __tablename__ = 'knowledge_database'

    knowledge_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    category = Column(String)
    create_dt = Column(DateTime, default=datetime.utcnow)
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    documents = relationship("KnowledgeDocument", back_populates="knowledge")


class KnowledgeDocument(Base):
    __tablename__ = 'knowledge_document'

    document_id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String)
    category = Column(String)
    knowledge_id = Column(Integer, ForeignKey('knowledge_database.knowledge_id'))
    file_path = Column(String)
    file_type = Column(String)
    create_dt = Column(DateTime, default=datetime.utcnow)
    update_dt = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    knowledge = relationship("KnowledgeDatabase", back_populates="documents")


Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
