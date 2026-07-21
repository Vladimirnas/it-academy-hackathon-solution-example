"""Загрузка чата и размеченных вопросов в память."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))

PROTECTED = {"Go Nova.json", "demo_chat.json"}
DEFAULT_CHAT = os.getenv("CHAT_FILE", "Go Nova.json")
QUESTIONS_FILE = Path(os.getenv("QUESTIONS_FILE", "/app/questions.json"))


def chat_path(name: str) -> Path | None:
    candidate = DATA_DIR / name
    return candidate if candidate.is_file() else None


def available_chats() -> list[str]:
    if not DATA_DIR.is_dir():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.json"))


def render_message(message: dict, include_quotes: bool = True) -> str:
    parts: list[str] = []
    if message.get("text"):
        parts.append(message["text"])

    for part in message.get("parts") or []:
        text = part.get("text")
        if not isinstance(text, str) or not text:
            continue
        media = part.get("mediaType")
        if media == "quote" and not include_quotes:
            continue
        label = {"quote": "цитата", "forward": "переслано"}.get(media)
        parts.append(f"[{label}] {text}" if label else text)

    return "\n".join(parts)


class Corpus:

    def __init__(self, chat_file: str = DEFAULT_CHAT, questions_file: Path = QUESTIONS_FILE):
        path = chat_path(chat_file)
        if path is None:
            raise FileNotFoundError(f"чат не найден: {chat_file}")
        self.file_name = chat_file

        with open(path, encoding="utf-8") as f:
            raw = json.load(f)

        self.chat: dict = raw["chat"]
        self.raw_messages: list[dict] = raw["messages"]
        self.messages: dict[str, dict] = {
            m["id"]: {
                "id": m["id"],
                "sender": m.get("sender_id", ""),
                "date": datetime.fromtimestamp(
                    m.get("time", 0), tz=timezone.utc
                ).strftime("%d.%m.%Y"),
                "text": render_message(m),
                "text_llm": render_message(m, include_quotes=False),
                "is_forward": bool(m.get("is_forward")),
                "is_quote": bool(m.get("is_quote")),
            }
            for m in raw["messages"]
        }

        if questions_file.exists():
            with open(questions_file, encoding="utf-8") as f:
                self.questions: list[dict] = json.load(f)
        else:
            self.questions = []

    def gold_ids_for(self, question_text: str) -> list[str]:
        needle = question_text.strip()
        for q in self.questions:
            if q["question"]["text"].strip() == needle:
                return q["gold_message_ids"]
        return []

    def resolve(self, message_ids: list[str], gold_ids: list[str] | None = None) -> list[dict]:
        gold = set(gold_ids or [])
        resolved = []
        for mid in message_ids:
            message = self.messages.get(mid)
            if message is None:
                continue
            resolved.append({**message, "is_gold": mid in gold})
        return resolved

    def question_texts(self) -> list[str]:
        return [q["question"]["text"] for q in self.questions]
