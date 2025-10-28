from datetime import datetime
from typing import Any, Optional

import numpy as np
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from pydantic import InstanceOf, TypeAdapter, validate_call
from sqlalchemy import (
    Boolean,
    DateTime,
    Engine,
    ForeignKey,
    Integer,
    String,
    select,
)
from sqlalchemy.orm import DeclarativeBase, mapped_column, sessionmaker
from sqlalchemy.sql import func

from memmachine.semantic_memory.semantic_model import SemanticFeature, SemanticHistory
from memmachine.semantic_memory.storage.storage_base import SemanticStorageBase


class BaseSemanticStorage(DeclarativeBase):
    pass


class Feature(BaseSemanticStorage):
    __tablename__ = "semantic_feature"
    id = mapped_column(Integer, primary_key=True)

    # Feature data
    set_id = mapped_column(String)
    semantic_type_id = mapped_column(String)
    tag_id = mapped_column(String)
    feature = mapped_column(String)
    value = mapped_column(String)

    # metadata
    created_a = mapped_column(DateTime, server_default=func.now())
    updated_at = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())
    embedding = mapped_column(Vector)
    json_metadata = mapped_column(String, name="metadata", server_default="{}")

    def to_typed_model(self) -> SemanticFeature:
        return SemanticFeature(
            metadata=SemanticFeature.Metadata(
                id=self.id,
            ),
            set_id=self.set_id,
            type=self.semantic_type_id,
            tag=self.tag_id,
            feature=self.feature,
            value=self.value,
        )


class History(BaseSemanticStorage):
    __tablename__ = "history"
    id = mapped_column(Integer, primary_key=True)

    content = mapped_column(String)

    json_metadata = mapped_column(String, name="metadata", server_default="{}")
    created_at = mapped_column(DateTime, server_default=func.now())

    # semantic memory specific columns
    set_id = mapped_column(String)
    ingested = mapped_column(Boolean, default=False)

    def to_typed_model(self) -> SemanticHistory:
        return SemanticHistory(
            metadata=SemanticHistory.Metadata(
                id=self.id,
            ),
            set_id=self.set_id,
            content=self.content,
            created_at=self.created_at,
            ingested=self.ingested,
        )


class SemanticCitation(BaseSemanticStorage):
    __tablename__ = "semantic_citation"
    semantic_feature_id = mapped_column(
        Integer, ForeignKey("semantic_feature.id"), primary_key=True
    )
    history_id = mapped_column(Integer, ForeignKey("history.id"), primary_key=True)


class SqlAlchemyPgVectorSemanticStorage(SemanticStorageBase):
    def __init__(self, sqlalchemy_engine: Engine):
        self._engine = sqlalchemy_engine

    def _create_session(self):
        return sessionmaker(bind=self._engine)()

    def _initialize_db(self):
        with self._engine.begin() as conn:
            conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")

        BaseSemanticStorage.metadata.create_all(self._engine)

    async def startup(self):
        self._initialize_db()

    async def cleanup(self):
        pass

    @validate_call
    async def delete_all(self):
        with self._create_session() as session:
            session.query(Feature).delete()
            session.query(History).delete()
            session.query(SemanticCitation).delete()
            session.commit()

    @validate_call
    async def add_feature(
        self,
        *,
        set_id: str,
        semantic_type_id: str,
        feature: str,
        value: str,
        tag: str,
        embedding: InstanceOf[np.ndarray],
        metadata: dict[str, Any] | None = None,
        citations: list[int] | None = None,
    ) -> int:
        stmt = (
            sa.insert(Feature)
            .values(
                set_id=set_id,
                semantic_type_id=semantic_type_id,
                tag_id=tag,
                feature=feature,
                value=value,
                embedding=embedding,
                json_metadata=metadata,
            )
            .returning(Feature.id)
        )

        with self._create_session() as session:
            res = session.execute(stmt).scalar_one()
            session.commit()

        return res

    @validate_call
    async def get_feature(self, feature_id: int) -> SemanticFeature | None:
        stmt = sa.select(Feature).where(Feature.id == feature_id)

        with self._create_session() as session:
            return session.execute(stmt).scalar_one_or_none()

    @validate_call
    async def get_feature_set(
        self,
        *,
        set_id: Optional[str] = None,
        semantic_type_id: Optional[str] = None,
        feature_name: Optional[str] = None,
        tag: Optional[str] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
        thresh: Optional[int] = None,
    ) -> list[SemanticFeature]:
        stmt = sa.select(Feature)

        stmt = self._apply_feature_filter(
            stmt,
            set_id=set_id,
            semantic_type_id=semantic_type_id,
            tag=tag,
            feature_name=feature_name,
            thresh=thresh,
            k=k,
            vector_search_opts=vector_search_opts,
        )

        with self._create_session() as session:
            features = session.execute(stmt).scalars().all()

        return [f.to_typed_model() for f in features]

    @validate_call
    async def delete_features(self, feature_ids: list[int]):
        stmt = sa.delete(Feature).where(Feature.id.in_(feature_ids))
        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def delete_feature_set(
        self,
        *,
        set_id: Optional[str] = None,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
        feature_name: Optional[str] = None,
        thresh: Optional[int] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
    ):
        stmt = sa.delete(Feature)

        stmt = self._apply_feature_filter(
            stmt,
            set_id=set_id,
            semantic_type_id=semantic_type_id,
            tag=tag,
            feature_name=feature_name,
            thresh=thresh,
            k=k,
            vector_search_opts=vector_search_opts,
        )

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def get_all_citations_for_ids(self, feature_ids: list[int]) -> list[int]:
        pass

    @validate_call
    async def add_history(
        self,
        set_id: str,
        content: str,
        metadata: dict[str, str] | None = None,
        created_at: Optional[datetime] = None,
    ) -> int:
        stmt = (
            sa.insert(History)
            .values(
                set_id=set_id,
                content=content,
                # json_metadata=metadata,
            )
            .returning(History.id)
        )

        if created_at is not None:
            stmt = stmt.values(created_at=created_at)

        with self._create_session() as session:
            res = session.execute(stmt).scalar_one()
            session.commit()

        return res

    @validate_call
    async def get_history(self, history_id: int) -> Optional[dict[str, Any]]:
        stmt = sa.select(History).where(History.id == history_id)

        with self._create_session() as session:
            return session.execute(stmt).scalar_one_or_none()

    @validate_call
    async def delete_history(self, history_id: int):
        stmt = sa.delete(History).where(History.id == history_id)

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def delete_history_messages(
        self,
        *,
        set_id: Optional[str] = None,
        k: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        is_ingested: Optional[bool] = None,
    ):
        stmt = sa.delete(History)

        stmt = self._apply_history_filter(
            stmt,
            set_id=set_id,
            start_time=start_time,
            end_time=end_time,
            k=k,
            is_ingested=is_ingested,
        )

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @validate_call
    async def get_history_messages(
        self,
        *,
        set_id: Optional[str] = None,
        k: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        is_ingested: Optional[bool] = None,
    ) -> list[SemanticHistory]:
        stmt = sa.select(History)

        # Order oldest to newest [ first, second, third ]
        stmt = stmt.order_by(History.created_at.asc())

        stmt = self._apply_history_filter(
            stmt,
            set_id=set_id,
            start_time=start_time,
            end_time=end_time,
            k=k,
            is_ingested=is_ingested,
        )

        with self._create_session() as session:
            history_messages = session.execute(stmt).scalars().all()

        return [h.to_typed_model() for h in history_messages]

    @validate_call
    async def get_history_messages_count(
        self,
        *,
        set_id: Optional[str] = None,
        k: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        is_ingested: Optional[bool] = None,
    ) -> int:
        stmt = sa.select(func.count(History.id))

        stmt = self._apply_history_filter(
            stmt,
            set_id=set_id,
            start_time=start_time,
            end_time=end_time,
            k=k,
            is_ingested=is_ingested,
        )

        with self._create_session() as session:
            return session.execute(stmt).scalar_one()

    @validate_call
    async def mark_messages_ingested(self, ids: list[int]) -> None:
        if len(ids) == 0:
            raise ValueError("No ids provided")

        stmt = sa.update(History).where(History.id.in_(ids)).values(ingested=True)

        with self._create_session() as session:
            session.execute(stmt)
            session.commit()

    @staticmethod
    def _apply_history_filter(
        stmt,
        *,
        set_id: Optional[str] = None,
        k: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        is_ingested: Optional[bool] = None,
    ):

        if set_id is not None:
            stmt = stmt.where(History.set_id == set_id)
        if start_time is not None:
            stmt = stmt.where(History.created_at >= start_time)
        if end_time is not None:
            stmt = stmt.where(History.created_at <= end_time)
        if is_ingested is not None:
            stmt = stmt.where(History.ingested == is_ingested)
        if k is not None:
            stmt = stmt.limit(k)

        return stmt

    def _apply_feature_filter(
        self,
        stmt,
        *,
        set_id: Optional[str] = None,
        semantic_type_id: Optional[str] = None,
        tag: Optional[str] = None,
        feature_name: Optional[str] = None,
        thresh: Optional[int] = None,
        k: Optional[int] = None,
        vector_search_opts: Optional[SemanticStorageBase.VectorSearchOpts] = None,
    ):
        def _apply_feature_id_filter(
            _stmt,
        ):
            if set_id is not None:
                _stmt = _stmt.where(Feature.set_id == set_id)
            if semantic_type_id is not None:
                _stmt = _stmt.where(Feature.semantic_type_id == semantic_type_id)
            if tag is not None:
                _stmt = _stmt.where(Feature.tag_id == tag)
            if feature_name is not None:
                _stmt = _stmt.where(Feature.feature == feature_name)
            if k is not None:
                _stmt = _stmt.limit(k)
            if vector_search_opts is not None:
                _stmt = _stmt.where(
                    Feature.embedding.cosine_distance(
                        vector_search_opts.query_embedding
                    )
                    > vector_search_opts.min_cos
                ).order_by(
                    Feature.embedding.cosine_distance(
                        vector_search_opts.query_embedding
                    ).desc()
                )

            return _stmt

        stmt = _apply_feature_id_filter(stmt)

        if thresh is not None:
            subquery = self._get_tags_with_more_than_k_features(thresh)
            subquery = _apply_feature_id_filter(subquery)

            stmt = stmt.where(Feature.tag_id.in_(subquery))

        return stmt

    @staticmethod
    def _get_tags_with_more_than_k_features(k: int):
        return select(Feature.tag_id).group_by(Feature.tag_id).having(func.count() >= k)
