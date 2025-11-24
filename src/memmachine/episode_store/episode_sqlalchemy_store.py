"""SQLAlchemy implementation of the episode storage layer."""

from collections.abc import Callable
from datetime import UTC
from typing import Any, TypeVar

from pydantic import AwareDatetime, JsonValue, validate_call
from sqlalchemy import (
    JSON,
    DateTime,
    Delete,
    Index,
    Integer,
    String,
    and_,
    cast,
    delete,
    func,
    insert,
    or_,
    select,
)
from sqlalchemy import (
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy.sql import Select
from sqlalchemy.sql.elements import ColumnElement

from memmachine.common.filter.filter_parser import (
    And as FilterAnd,
)
from memmachine.common.filter.filter_parser import (
    Comparison as FilterComparison,
)
from memmachine.common.filter.filter_parser import (
    FilterExpr,
)
from memmachine.common.filter.filter_parser import (
    Or as FilterOr,
)
from memmachine.episode_store.episode_model import (
    Episode as EpisodeE,
)
from memmachine.episode_store.episode_model import EpisodeEntry, EpisodeType
from memmachine.episode_store.episode_storage import EpisodeIdT, EpisodeStorage


class BaseEpisodeStore(DeclarativeBase):
    """Base class for SQLAlchemy Episode store."""


JSON_AUTO = JSON().with_variant(JSONB, "postgresql")

StmtT = TypeVar("StmtT", Select[Any], Delete)


class Episode(BaseEpisodeStore):
    """SQLAlchemy mapping for stored conversation messages."""

    __tablename__ = "episodestore"
    id = mapped_column(Integer, primary_key=True)

    content = mapped_column(String, nullable=False)

    session_key = mapped_column(String, nullable=False)
    producer_id = mapped_column(String, nullable=False)
    producer_role = mapped_column(String, nullable=False)

    produced_for_id = mapped_column(String, nullable=True)
    episode_type = mapped_column(
        SAEnum(EpisodeType, name="episode_type"),
        default=EpisodeType.MESSAGE,
    )

    json_metadata = mapped_column(
        JSON_AUTO,
        name="metadata",
        default=dict,
        nullable=False,
    )
    created_at = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    __table_args__ = (
        Index("idx_session_key", "session_key"),
        Index("idx_producer_id", "producer_id"),
        Index("idx_producer_role", "producer_role"),
        Index("idx_session_key_producer_id", "session_key", "producer_id"),
        Index(
            "idx_session_key_producer_id_producer_role_produced_for_id",
            "session_key",
            "producer_id",
            "producer_role",
            "produced_for_id",
        ),
    )

    def to_typed_model(self) -> EpisodeE:
        created_at = (
            self.created_at.replace(tzinfo=UTC)
            if self.created_at.tzinfo is None
            else self.created_at
        )
        return EpisodeE(
            uid=EpisodeIdT(self.id),
            content=self.content,
            session_key=self.session_key,
            producer_id=self.producer_id,
            producer_role=self.producer_role,
            produced_for_id=self.produced_for_id,
            episode_type=self.episode_type,
            created_at=created_at,
            metadata=self.json_metadata or None,
        )


class SqlAlchemyEpisodeStore(EpisodeStorage):
    """SQLAlchemy episode store implementation."""

    def __init__(self, engine: AsyncEngine) -> None:
        """Initialize the store with an async SQLAlchemy engine."""
        self._engine: AsyncEngine = engine
        self._session_factory = async_sessionmaker(
            self._engine,
            expire_on_commit=False,
        )

    def _create_session(self) -> AsyncSession:
        return self._session_factory()

    async def startup(self) -> None:
        async with self._engine.begin() as conn:
            await conn.run_sync(BaseEpisodeStore.metadata.create_all)

    @validate_call
    async def add_episode(
        self,
        *,
        content: str,
        session_key: str,
        producer_id: str,
        producer_role: str,
        produced_for_id: str | None = None,
        episode_type: EpisodeType | None = None,
        metadata: dict[str, JsonValue] | None = None,
        created_at: AwareDatetime | None = None,
    ) -> EpisodeIdT:
        stmt = (
            insert(Episode)
            .values(
                content=content,
                session_key=session_key,
                producer_id=producer_id,
                producer_role=producer_role,
            )
            .returning(Episode.id)
        )

        if produced_for_id is not None:
            stmt = stmt.values(produced_for_id=produced_for_id)

        if episode_type is not None:
            stmt = stmt.values(episode_type=episode_type)

        if metadata is not None:
            stmt = stmt.values(json_metadata=metadata)

        if created_at is not None:
            stmt = stmt.values(created_at=created_at)

        async with self._create_session() as session:
            result = await session.execute(stmt)
            await session.commit()
            episode_id = result.scalar_one()

        return EpisodeIdT(episode_id)

    @validate_call
    async def add_episodes(
        self,
        session_key: str,
        episodes: list[EpisodeEntry],
    ) -> list[EpisodeE]:
        if not episodes:
            return []

        values_to_insert: list[dict[str, Any]] = []
        for entry in episodes:
            entry_values: dict[str, Any] = {
                "content": entry.content,
                "session_key": session_key,
                "producer_id": entry.producer_id,
                "producer_role": entry.producer_role,
            }

            if entry.produced_for_id is not None:
                entry_values["produced_for_id"] = entry.produced_for_id

            if entry.episode_type is not None:
                entry_values["episode_type"] = entry.episode_type

            if entry.metadata is not None:
                entry_values["json_metadata"] = entry.metadata

            if entry.created_at is not None:
                entry_values["created_at"] = entry.created_at

            values_to_insert.append(entry_values)

        insert_stmt = insert(Episode).returning(Episode.id)

        async with self._create_session() as session:
            result = await session.execute(insert_stmt, values_to_insert)
            inserted_ids = result.scalars().all()
            await session.commit()

        int_episode_ids = [int(episode_id) for episode_id in inserted_ids]
        if not int_episode_ids:
            return []

        select_stmt = (
            select(Episode)
            .where(Episode.id.in_(int_episode_ids))
            .order_by(Episode.id.asc())
        )

        async with self._create_session() as session:
            result = await session.execute(select_stmt)
            persisted_episodes = result.scalars().all()

        episodes_by_id = {
            episode_row.id: episode_row.to_typed_model()
            for episode_row in persisted_episodes
        }
        return [episodes_by_id[episode_id] for episode_id in int_episode_ids]

    @validate_call
    async def get_episode(self, episode_id: EpisodeIdT) -> EpisodeE | None:
        stmt = (
            select(Episode)
            .where(Episode.id == int(episode_id))
            .order_by(Episode.created_at.asc())
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            episode = result.scalar_one_or_none()

        return episode.to_typed_model() if episode else None

    def _apply_episode_filter(
        self,
        stmt: StmtT,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> StmtT:
        filters: list[ColumnElement[bool]] = []

        if filter_expr is not None:
            filters.append(self._build_filter_clause(filter_expr))
        if start_time is not None:
            filters.append(Episode.created_at >= start_time)
        if end_time is not None:
            filters.append(Episode.created_at <= end_time)

        if filters:
            stmt = stmt.where(*filters)

        return stmt

    def _build_filter_clause(self, expr: FilterExpr) -> ColumnElement[bool]:
        if isinstance(expr, FilterComparison):
            return self._build_comparison_clause(expr)
        if isinstance(expr, FilterAnd):
            return and_(
                self._build_filter_clause(expr.left),
                self._build_filter_clause(expr.right),
            )
        if isinstance(expr, FilterOr):
            return or_(
                self._build_filter_clause(expr.left),
                self._build_filter_clause(expr.right),
            )
        raise TypeError(f"Unsupported filter expression type: {type(expr)!r}")

    def _build_comparison_clause(
        self,
        comparison: FilterComparison,
    ) -> ColumnElement[bool]:
        field = comparison.field
        if field.startswith(("metadata.", "m.")):
            key = field.split(".", 1)[1]
            return self._build_metadata_comparison(key, comparison)

        column, converter = self._resolve_basic_field(field)
        return self._build_basic_comparison(column, converter, comparison)

    def _build_basic_comparison(
        self,
        column: ColumnElement[Any],
        converter: Callable[[Any], Any] | None,
        comparison: FilterComparison,
    ) -> ColumnElement[bool]:
        op = comparison.op
        if op == "=":
            value = self._normalize_value(comparison.value, converter)
            if value is None:
                return column.is_(None)
            return column == value
        if op == "in":
            if not isinstance(comparison.value, list):
                raise TypeError("IN comparison requires a list of values")
            values = [
                self._normalize_value(value, converter)
                for value in comparison.value
            ]
            null_selected = any(value is None for value in values)
            non_null_values = [value for value in values if value is not None]
            clauses: list[ColumnElement[bool]] = []
            if non_null_values:
                clauses.append(column.in_(non_null_values))
            if null_selected:
                clauses.append(column.is_(None))
            if not clauses:
                return column.is_(None)  # pragma: no cover
            if len(clauses) == 1:
                return clauses[0]
            return or_(*clauses)
        if op == "is_null":
            return column.is_(None)
        if op == "is_not_null":
            return column.is_not(None)
        raise ValueError(f"Unsupported operator for field '{comparison.field}': {op}")

    def _build_metadata_comparison(
        self,
        key: str,
        comparison: FilterComparison,
    ) -> ColumnElement[bool]:
        op = comparison.op
        if op == "=":
            return self._metadata_equality_clause(key, [comparison.value])
        if op == "in":
            if not isinstance(comparison.value, list):
                raise TypeError("IN comparison requires a list of values")
            return self._metadata_equality_clause(key, comparison.value)
        if op == "is_null":
            value_expr = self._metadata_value_expr(key)
            return value_expr.is_(None)
        if op == "is_not_null":
            value_expr = self._metadata_value_expr(key)
            return value_expr.is_not(None)
        raise ValueError(f"Unsupported operator for metadata field '{key}': {op}")

    def _metadata_equality_clause(
        self,
        key: str,
        raw_values: list[Any],
    ) -> ColumnElement[bool]:
        clauses: list[ColumnElement[bool]] = []
        for raw_value in raw_values:
            if raw_value is None:
                clauses.append(self._metadata_value_expr(key).is_(None))
                continue
            contains_clause = self._metadata_contains_clause(key, raw_value)
            clauses.append(contains_clause)
        if not clauses:
            return self._metadata_value_expr(key).is_(None)
        if len(clauses) == 1:
            return clauses[0]
        return or_(*clauses)

    def _metadata_contains_clause(
        self,
        key: str,
        raw_value: Any,
    ) -> ColumnElement[bool]:
        value = self._normalize_metadata_value(raw_value)
        metadata_dict = self._build_metadata_dict(key, value)
        if self._engine.dialect.name == "postgresql":
            return cast(Episode.json_metadata, JSONB).contains(metadata_dict)
        return Episode.json_metadata.contains(metadata_dict)

    def _metadata_value_expr(self, key: str) -> ColumnElement[Any]:
        path = key.split(".")
        if self._engine.dialect.name == "postgresql":
            expr: ColumnElement[Any] = cast(Episode.json_metadata, JSONB)
            for part in path:
                expr = expr[part]
            return expr.astext
        json_path = "$." + ".".join(path)
        return func.json_extract(Episode.json_metadata, json_path)

    @staticmethod
    def _build_metadata_dict(key: str, value: Any) -> dict[str, Any]:
        parts = key.split(".")
        current: dict[str, Any] = {}
        root = current
        for part in parts[:-1]:
            next_level: dict[str, Any] = {}
            current[part] = next_level
            current = next_level
        current[parts[-1]] = value
        return root

    def _resolve_basic_field(
        self,
        field: str,
    ) -> tuple[ColumnElement[Any], Callable[[Any], Any] | None]:
        mapping: dict[str, tuple[ColumnElement[Any], Callable[[Any], Any] | None]] = {
            "session_key": (Episode.session_key, self._coerce_str),
            "session": (Episode.session_key, self._coerce_str),
            "producer_id": (Episode.producer_id, self._coerce_str),
            "producer_role": (Episode.producer_role, self._coerce_str),
            "role": (Episode.producer_role, self._coerce_str),
            "produced_for_id": (Episode.produced_for_id, self._coerce_str),
            "episode_type": (Episode.episode_type, self._normalize_episode_type),
            "type": (Episode.episode_type, self._normalize_episode_type),
            "id": (Episode.id, self._normalize_episode_id),
            "uid": (Episode.id, self._normalize_episode_id),
            "episode_id": (Episode.id, self._normalize_episode_id),
        }
        if field not in mapping:
            raise ValueError(f"Unsupported episode filter field: {field}")
        return mapping[field]

    @staticmethod
    def _coerce_str(value: Any) -> str | None:
        if value is None:
            return None
        return str(value)

    @staticmethod
    def _normalize_value(
        value: Any,
        converter: Callable[[Any], Any] | None,
    ) -> Any:
        if value is None:
            return None
        if converter is None:
            return value
        return converter(value)

    def _normalize_metadata_value(self, value: Any) -> Any:
        if value is None:
            return None
        if self._engine.dialect.name == "postgresql":
            return value
        return str(value)

    @staticmethod
    def _normalize_episode_type(value: Any) -> EpisodeType:
        if isinstance(value, EpisodeType):
            return value
        if isinstance(value, str):
            name = value.upper()
            if name in EpisodeType.__members__:
                return EpisodeType[name]
            try:
                return EpisodeType(value.lower())
            except ValueError as exc:  # pragma: no cover - defensive
                raise ValueError(f"Unknown episode type: {value}") from exc
        raise TypeError("Episode type filters must be strings or EpisodeType values")

    @staticmethod
    def _normalize_episode_id(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
            raise ValueError("Episode id filters must be numeric") from exc

    async def get_episode_messages(
        self,
        *,
        limit: int | None = None,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> list[EpisodeE]:
        stmt = select(Episode)

        stmt = self._apply_episode_filter(
            stmt,
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

        if limit is not None:
            stmt = stmt.limit(limit)

        async with self._create_session() as session:
            result = await session.execute(stmt)
            episode_messages = result.scalars().all()

        return [h.to_typed_model() for h in episode_messages]

    async def get_episode_messages_count(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> int:
        stmt = select(func.count(Episode.id))

        stmt = self._apply_episode_filter(
            stmt,
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

        async with self._create_session() as session:
            result = await session.execute(stmt)
            n_messages = result.scalar_one()

        return int(n_messages)

    @validate_call
    async def delete_episodes(self, episode_ids: list[EpisodeIdT]) -> None:
        int_episode_ids = [int(h_id) for h_id in episode_ids]

        stmt = delete(Episode).where(Episode.id.in_(int_episode_ids))

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()

    async def delete_episode_messages(
        self,
        *,
        filter_expr: FilterExpr | None = None,
        start_time: AwareDatetime | None = None,
        end_time: AwareDatetime | None = None,
    ) -> None:
        stmt = delete(Episode)

        stmt = self._apply_episode_filter(
            stmt,
            filter_expr=filter_expr,
            start_time=start_time,
            end_time=end_time,
        )

        async with self._create_session() as session:
            await session.execute(stmt)
            await session.commit()
