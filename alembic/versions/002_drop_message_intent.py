"""Drop unused messages.intent column.

Revision ID: 002
Revises: 001
Create Date: 2026-06-26
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("messages", "intent")


def downgrade() -> None:
    op.add_column("messages", sa.Column("intent", sa.String(length=16), nullable=True))
