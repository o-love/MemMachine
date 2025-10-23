from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.sql import func

from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


class BaseSemanticStorage(DeclarativeBase):
    pass


class Feature(BaseSemanticStorage):
    __tablename__ = "semantic_feature"
    id = Column(Integer, primary_key=True)

    # Feature data
    set_id = Column(String)
    semantic_type_id = Column(String)
    tag = Column(String)
    feature = Column(String)
    value = Column(String)

    # metadata
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    embedding = Column(Vector(1536))
    metadata = Column(String, server_default="{}")


class History(BaseSemanticStorage):
    __tablename__ = "history"
    id = Column(Integer, primary_key=True)

    content = Column(String)

    metadata = Column(String, server_default="{}")
    created_at = Column(DateTime, server_default=func.now())

    # semantic memory specific columns
    set_id = Column(String)
    ingested = Column(Boolean, default=False)


class SemanticCitation(BaseSemanticStorage):
    __tablename__ = "semantic_citation"
    semantic_feature_id = Column(
        Integer, ForeignKey("semantic_feature.id"), primary_key=True
    )
    history_id = Column(Integer, ForeignKey("history.id"), primary_key=True)


class SqlAlchemyPgVectorSemanticStorage(SemanticStorageBase):
    def __init__(self, sqlalchemy_engine: Engine):
        self._engine = sqlalchemy_engine

    def _create_session(self):
        return sessionmaker(bind=self._engine)()

    def _initialize_db(self):
        BaseSemanticStorage.metadata.create_all(self._engine)

    def _get_feature_by_id(self, feature_id: int):
        with self._create_session() as session:
            return session.query(Feature).filter(Feature.id == feature_id).first()
