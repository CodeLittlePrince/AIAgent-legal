"""API 请求与响应的数据模型（Schema）。

使用 Pydantic BaseModel 定义接口的输入输出结构，
FastAPI 会自动据此做 JSON 解析、校验和 OpenAPI 文档生成。
"""

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """法律回答中的引用来源。

    Attributes:
        source: 文档或法规名称。
        excerpt: 引用的原文片段。
    """

    source: str
    excerpt: str


class ChatRequest(BaseModel):
    """POST /api/v1/chat 的请求体。

    Attributes:
        session_id: 会话 ID；不传则服务端自动生成新会话。
        message: 用户输入的消息，至少 1 个字符。
    """

    session_id: str | None = None
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    """POST /api/v1/chat 的响应体。

    Attributes:
        session_id: 本次对话所属的会话 ID。
        intent: 识别出的意图（legal / weather / general）。
        answer: 助手生成的回答正文。
        citations: 法律类回答附带的引用列表。
        disclaimer: 免责声明；仅 legal 意图时可能有值。
        trace_id: 链路追踪 ID，用于反馈评分。
    """

    session_id: str
    intent: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    disclaimer: str | None = None
    trace_id: str


class FeedbackRequest(BaseModel):
    """POST /api/v1/feedback 的请求体。

    Attributes:
        trace_id: 要评分的对话 trace ID。
        score: 用户评分，0 或 1（不满意 / 满意）。
        comment: 可选的文字反馈。
    """

    trace_id: str
    score: int = Field(ge=0, le=1)
    comment: str | None = None


class SessionResponse(BaseModel):
    """GET /api/v1/sessions/{session_id} 的响应体。

    Attributes:
        session_id: 会话 ID。
        messages: 该会话的历史消息列表，每条为 {"role": ..., "content": ...} 格式。
    """

    session_id: str
    messages: list[dict]
