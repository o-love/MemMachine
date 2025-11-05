"""sync_sqlalchemy_schema

Revision ID: 3d6aaebdc526
Revises: 001
Create Date: 2025-11-04 20:32:38.622715

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3d6aaebdc526"
down_revision: Union[str, Sequence[str], None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema from legacy prof/history tables to SQLAlchemy layout."""
    # Drop old citations table (schema changes make it incompatible)
    op.execute("DROP TABLE IF EXISTS citations CASCADE")

    # Rename prof table to feature and align columns
    op.execute("ALTER TABLE prof RENAME TO feature")
    op.execute("ALTER TABLE feature RENAME COLUMN user_id TO set_id")
    op.execute("ALTER TABLE feature RENAME COLUMN tag TO tag_id")
    op.execute("ALTER TABLE feature RENAME COLUMN create_at TO created_at")
    op.execute("ALTER TABLE feature RENAME COLUMN update_at TO updated_at")
    op.execute(
        "ALTER TABLE feature ADD COLUMN semantic_type_id VARCHAR DEFAULT 'default'"
    )
    op.execute("ALTER TABLE feature DROP COLUMN IF EXISTS isolations")
    op.execute(
        "ALTER TABLE feature ALTER COLUMN created_at TYPE TIMESTAMP USING created_at::TIMESTAMP"
    )
    op.execute(
        "ALTER TABLE feature ALTER COLUMN updated_at TYPE TIMESTAMP USING updated_at::TIMESTAMP"
    )
    op.execute(
        "ALTER TABLE feature ALTER COLUMN metadata TYPE TEXT USING metadata::TEXT"
    )
    op.execute("ALTER TABLE feature ALTER COLUMN metadata SET DEFAULT '{}'::TEXT")

    # Rebuild feature indexes
    op.execute("DROP INDEX IF EXISTS prof_user_idx")
    op.create_index("idx_feature_set_id", "feature", ["set_id"])
    op.create_index(
        "idx_feature_set_id_semantic_type", "feature", ["set_id", "semantic_type_id"]
    )
    op.create_index(
        "idx_feature_set_semantic_type_tag",
        "feature",
        ["set_id", "semantic_type_id", "tag_id"],
    )
    op.create_index(
        "idx_feature_set_semantic_type_tag_feature",
        "feature",
        ["set_id", "semantic_type_id", "tag_id", "feature"],
    )

    # Align history table
    op.execute("ALTER TABLE history RENAME COLUMN create_at TO created_at")
    op.execute(
        "ALTER TABLE history ALTER COLUMN created_at TYPE TIMESTAMP USING created_at::TIMESTAMP"
    )
    op.execute(
        "ALTER TABLE history ALTER COLUMN metadata TYPE TEXT USING metadata::TEXT"
    )
    op.execute("ALTER TABLE history ALTER COLUMN metadata SET DEFAULT '{}'::TEXT")
    op.execute("ALTER TABLE history DROP COLUMN IF EXISTS user_id")
    op.execute("ALTER TABLE history DROP COLUMN IF EXISTS ingested")
    op.execute("ALTER TABLE history DROP COLUMN IF EXISTS isolations")
    op.execute("DROP INDEX IF EXISTS history_user_idx")
    op.execute("DROP INDEX IF EXISTS history_user_ingested_idx")
    op.execute("DROP INDEX IF EXISTS history_user_ingested_ts_desc")

    # Create set_ingested_history helper table
    op.create_table(
        "set_ingested_history",
        sa.Column("set_id", sa.String(), nullable=False),
        sa.Column("history_id", sa.Integer(), nullable=False),
        sa.Column(
            "ingested", sa.Boolean(), nullable=True, server_default=sa.text("false")
        ),
        sa.ForeignKeyConstraint(
            ["history_id"], ["history.id"], ondelete="CASCADE", onupdate="CASCADE"
        ),
        sa.PrimaryKeyConstraint("set_id", "history_id"),
    )

    # Recreate citations table with new structure
    op.create_table(
        "citations",
        sa.Column("feature_id", sa.Integer(), nullable=False),
        sa.Column("history_id", sa.Integer(), nullable=False),
        sa.ForeignKeyConstraint(
            ["feature_id"], ["feature.id"], ondelete="CASCADE", onupdate="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["history_id"], ["history.id"], ondelete="CASCADE", onupdate="CASCADE"
        ),
        sa.PrimaryKeyConstraint("feature_id", "history_id"),
    )


def downgrade() -> None:
    """Downgrade schema back to legacy layout."""
    op.execute("DROP TABLE IF EXISTS citations CASCADE")
    op.execute("DROP TABLE IF EXISTS set_ingested_history CASCADE")

    # Restore history table
    op.execute(
        "ALTER TABLE history ALTER COLUMN metadata TYPE JSONB USING metadata::JSONB"
    )
    op.execute("ALTER TABLE history ALTER COLUMN metadata SET DEFAULT '{}'::JSONB")
    op.execute(
        "ALTER TABLE history ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at::TIMESTAMPTZ"
    )
    op.execute("ALTER TABLE history RENAME COLUMN created_at TO create_at")
    op.execute("ALTER TABLE history ADD COLUMN user_id TEXT NOT NULL DEFAULT 'unknown'")
    op.execute("ALTER TABLE history ADD COLUMN ingested BOOLEAN NOT NULL DEFAULT FALSE")
    op.execute("ALTER TABLE history ADD COLUMN isolations JSONB NOT NULL DEFAULT '{}'")
    op.create_index("history_user_idx", "history", ["user_id"])
    op.create_index("history_user_ingested_idx", "history", ["user_id", "ingested"])
    op.execute(
        "CREATE INDEX history_user_ingested_ts_desc ON history (user_id, ingested, create_at DESC)"
    )

    # Restore feature table
    op.execute(
        "ALTER TABLE feature ALTER COLUMN metadata TYPE JSONB USING metadata::JSONB"
    )
    op.execute("ALTER TABLE feature ALTER COLUMN metadata SET DEFAULT '{}'::JSONB")
    op.execute(
        "ALTER TABLE feature ALTER COLUMN updated_at TYPE TIMESTAMPTZ USING updated_at::TIMESTAMPTZ"
    )
    op.execute(
        "ALTER TABLE feature ALTER COLUMN created_at TYPE TIMESTAMPTZ USING created_at::TIMESTAMPTZ"
    )
    op.execute("DROP INDEX IF EXISTS idx_feature_set_semantic_type_tag_feature")
    op.execute("DROP INDEX IF EXISTS idx_feature_set_semantic_type_tag")
    op.execute("DROP INDEX IF EXISTS idx_feature_set_id_semantic_type")
    op.execute("DROP INDEX IF EXISTS idx_feature_set_id")
    op.execute("ALTER TABLE feature DROP COLUMN IF EXISTS semantic_type_id")
    op.execute("ALTER TABLE feature ADD COLUMN isolations JSONB NOT NULL DEFAULT '{}'")
    op.execute("ALTER TABLE feature RENAME COLUMN updated_at TO update_at")
    op.execute("ALTER TABLE feature RENAME COLUMN created_at TO create_at")
    op.execute("ALTER TABLE feature RENAME COLUMN tag_id TO tag")
    op.execute("ALTER TABLE feature RENAME COLUMN set_id TO user_id")
    op.execute("ALTER TABLE feature RENAME TO prof")
    op.create_index("prof_user_idx", "prof", ["user_id"])

    # Recreate legacy citations table
    op.execute(
        """
        CREATE TABLE citations (
            profile_id INTEGER REFERENCES prof(id) ON DELETE CASCADE,
            content_id INTEGER REFERENCES history(id) ON DELETE CASCADE,
            PRIMARY KEY (profile_id, content_id)
        )
        """
    )
