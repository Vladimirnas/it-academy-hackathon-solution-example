"""Индексирует чат в Qdrant через API backend.

Сама индексация живёт в orchestrator/main.py - здесь только вызов

    python3 eval/index_corpus.py                      # корпус для метрик
    python3 eval/index_corpus.py demo_chat.json       # другой чат из data/

Требует поднятого docker compose.
"""

import json
import sys
import urllib.error
import urllib.request

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND_URL = "http://localhost:8005"
DEFAULT_CHAT = "Go Nova.json"


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    chat_file = args[0] if args else DEFAULT_CHAT

    request = urllib.request.Request(
        f"{BACKEND_URL}/api/chat",
        data=json.dumps({"file": chat_file}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    print(f"Индексирую {chat_file}...")
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as e:
        detail = json.loads(e.read()).get("detail", e.reason)
        print(f"Ошибка: {detail}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"backend недоступен ({e.reason}). Поднят ли docker compose?", file=sys.stderr)
        sys.exit(1)

    print(f"Готово. Чат «{result['chat_name']}»: "
          f"{result['message_count']} сообщений, {result['chunks']} чанков.")


if __name__ == "__main__":
    main()
