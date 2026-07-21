"""Схемы запросов и ответов веб-интерфейса."""

from pydantic import BaseModel, Field


class Message(BaseModel):
    id: str
    sender: str
    date: str
    text: str
    is_forward: bool = False
    is_quote: bool = False
    is_gold: bool = False


class SearchRequest(BaseModel):
    text: str = Field(min_length=1, description="Вопрос на естественном языке")


class SearchResponse(BaseModel):
    messages: list[Message]
    gold_ids: list[str] = Field(
        default_factory=list,
        description="Ожидаемые ответы, если вопрос есть в размеченном наборе",
    )
    elapsed_ms: int
    llm_available: bool


class AnswerRequest(BaseModel):
    text: str = Field(min_length=1)
    message_ids: list[str] = Field(min_length=1)


class Source(BaseModel):
    n: int
    id: str
    sender: str
    date: str
    text: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[Source]
    model: str
    elapsed_ms: int


class ServiceState(BaseModel):
    ok: bool
    points: int | None = None
    error: str | None = None


class InfoResponse(BaseModel):
    chat_name: str
    message_count: int
    questions: list[str]
    chat_file: str
    available_chats: list[str] = Field(default_factory=list)


class SwitchChatRequest(BaseModel):
    file: str = Field(min_length=1, description="Имя файла чата в каталоге данных")


class SwitchChatResponse(BaseModel):
    chat_name: str
    chat_file: str
    message_count: int
    chunks: int


class UploadResponse(BaseModel):
    chat_name: str
    chat_file: str
    message_count: int
    available_chats: list[str]
