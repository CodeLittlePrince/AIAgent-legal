from pydantic import BaseModel, Field


class Citation(BaseModel):
    source: str
    excerpt: str


class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    session_id: str
    intent: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    disclaimer: str | None = None
    trace_id: str


class FeedbackRequest(BaseModel):
    trace_id: str
    score: int = Field(ge=0, le=1)
    comment: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    messages: list[dict]
