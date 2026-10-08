"""create original content table

Revision ID: 9b3d5d4c1a82
Revises: 432cf92453e4
Create Date: 2026-10-08

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9b3d5d4c1a82"
down_revision: Union[str, Sequence[str], None] = "432cf92453e4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "original_content",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("session_id", sa.Integer(), nullable=False),
        sa.Column("knowledge_id", sa.Integer(), nullable=False),
        sa.Column("entity_type", sa.String(length=20), nullable=False),
        sa.Column("value", sa.Text(), nullable=False),
        sa.ForeignKeyConstraint(["session_id"], ["sessions.id"]),
        sa.ForeignKeyConstraint(["knowledge_id"], ["knowledge.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_original_content_session_id",
        "original_content",
        ["session_id"],
        unique=False,
    )
    op.create_index(
        "ix_original_content_knowledge_id",
        "original_content",
        ["knowledge_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index("ix_original_content_knowledge_id", table_name="original_content")
    op.drop_index("ix_original_content_session_id", table_name="original_content")
    op.drop_table("original_content")
