"""Конвертация экспорта Telegram (result.json) в формат чатов проекта.
"""

import re


def slugify(name: str) -> str:
    return re.sub(r"[^\w-]+", "_", name.lower()).strip("_") or "telegram_chat"


CHAT_TYPES = {
    "private_supergroup": "group",
    "public_supergroup": "group",
    "private_group": "group",
    "public_channel": "channel",
    "private_channel": "channel",
    "personal_chat": "private",
    "saved_messages": "private",
}


def entity_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(
            part if isinstance(part, str) else part.get("text", "") for part in value
        )
    return ""


def extract_mentions(message: dict) -> list[str]:
    return sorted({
        e["text"].lstrip("@")
        for e in (message.get("text_entities") or [])
        if e.get("type", "").startswith("mention") and e.get("text")
    })


def file_note(message: dict) -> str:
    parts = []
    if message.get("media_type"):
        parts.append(message["media_type"])
    if message.get("file_name"):
        parts.append(message["file_name"])
    elif message.get("photo"):
        parts.append("фото")
    if message.get("sticker_emoji"):
        parts.append(f"стикер {message['sticker_emoji']}")
    return " ".join(parts)


def convert(raw: dict, limit: int, anonymize: bool) -> dict:
    source = [m for m in raw.get("messages", []) if m.get("type") == "message"]
    by_id = {m["id"]: m for m in source}

    if limit:
        source = source[-limit:]

    alias: dict[str, str] = {}

    def sender(message: dict) -> str:
        raw_id = str(message.get("from_id") or "unknown")
        if anonymize:
            if raw_id not in alias:
                alias[raw_id] = f"user{len(alias) + 1:03d}@corp.example"
            return alias[raw_id]
        name = message.get("from") or raw_id
        return f"{name}@telegram"

    messages = []
    for m in source:
        text = entity_text(m.get("text")).strip()
        note = file_note(m)
        if not text and not note:
            continue

        parts = []
        is_quote = False
        is_forward = bool(m.get("forwarded_from"))

        replied = by_id.get(m.get("reply_to_message_id"))
        if replied is not None:
            quoted = entity_text(replied.get("text")).strip()
            if quoted:
                parts.append({
                    "mediaType": "quote",
                    "sn": sender(replied),
                    "time": int(replied.get("date_unixtime", 0)),
                    "text": quoted[:1000],
                })
                is_quote = True

        if is_forward and text:
            parts.append({"mediaType": "forward", "sn": "", "time": 0, "text": text})

        messages.append({
            "id": str(m["id"]),
            "thread_sn": None,
            "time": int(m.get("date_unixtime", 0)),
            "text": "" if is_forward else text,
            "sender_id": sender(m),
            "file_snippets": note,
            "parts": parts,
            "mentions": [] if anonymize else extract_mentions(m),
            "member_event": None,
            "is_system": False,
            "is_hidden": False,
            "is_forward": is_forward,
            "is_quote": is_quote,
        })

    chat_id = str(raw.get("id", "0"))
    return {
        "chat": {
            "id": f"{chat_id}@chat.telegram",
            "name": raw.get("name") or "Telegram chat",
            "sn": f"{chat_id}@chat.telegram",
            "type": CHAT_TYPES.get(raw.get("type", ""), "group"),
            "is_public": raw.get("type", "").startswith("public"),
            "members_count": len({m["sender_id"] for m in messages}),
            "members": None,
        },
        "messages": messages,
    }
