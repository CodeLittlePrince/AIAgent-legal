import pytest

from legal_assistant.memory.models import Message, Session


def test_message_model_fields():
    assert "session_id" in Message.__table__.columns.keys()
    assert "role" in Message.__table__.columns.keys()
