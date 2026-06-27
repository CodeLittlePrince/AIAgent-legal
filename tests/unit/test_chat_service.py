import pytest

from legal_assistant.api.chat_service import chunk_text, format_sse, strip_trailing_disclaimer


def test_chunk_text_splits_answer():
    assert chunk_text("你好世界", size=2) == ["你好", "世界"]


def test_format_sse_event():
    payload = format_sse("delta", {"content": "试"})
    assert payload == 'event: delta\ndata: {"content": "试"}\n\n'


def test_strip_trailing_disclaimer():
    disclaimer = "本回答仅供参考，不构成法律意见。"
    answer = f"试用期最长六个月。\n\n{disclaimer}"
    assert strip_trailing_disclaimer(answer, disclaimer) == "试用期最长六个月。"


@pytest.mark.asyncio
async def test_iter_chat_sse_events_streams_with_delay():
    from legal_assistant.api.chat_service import ChatExecution, iter_chat_sse_events
    from legal_assistant.api.schemas import Citation

    execution = ChatExecution(
        session_id="s1",
        trace_id="t1",
        intent="legal",
        answer="你好",
        citations=[Citation(source="劳动法.md", excerpt="试用期…")],
        disclaimer="免责声明",
    )

    events = [
        event
        async for event in iter_chat_sse_events(execution, chunk_delay_seconds=0)
    ]
    body = "".join(events)

    assert body.index("event: citations") < body.index("event: delta")
    assert "event: disclaimer" in body
    assert "免责声明" not in body.split("event: delta")[1].split("event: disclaimer")[0]
