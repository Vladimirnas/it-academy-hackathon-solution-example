"""Загружает экспорт Telegram через API web-сервиса.



Экспорт делается в Telegram Desktop

    python3 eval/upload_telegram.py путь/к/result.json

Загруженный чат сохраняется в data/ и сразу становится активным.
Требует поднятого docker compose.


"""

import json
import mimetypes
import sys
import urllib.error
import urllib.request
import uuid
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BACKEND_URL = "http://localhost:8005"


def multipart_body(path: Path) -> tuple[bytes, str]:
    """Собирает multipart/form-data вручную - без сторонних библиотек."""
    boundary = uuid.uuid4().hex
    content_type = mimetypes.guess_type(path.name)[0] or "application/json"
    body = b"".join([
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'.encode(),
        f"Content-Type: {content_type}\r\n\r\n".encode(),
        path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ])
    return body, f"multipart/form-data; boundary={boundary}"


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    source = Path(sys.argv[1])
    if not source.is_file():
        print(f"Файл не найден: {source}", file=sys.stderr)
        sys.exit(1)

    body, content_type = multipart_body(source)
    request = urllib.request.Request(
        f"{BACKEND_URL}/api/upload",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )
    print(f"Загружаю {source.name} ({len(body) / 1048576:.1f} МБ)...")
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

    print(f"Готово. Чат «{result['chat_name']}»: {result['message_count']} сообщений")
    print(f"Сохранён как data/{result['chat_file']}")
    print(f"Проиндексировать: python3 eval/index_corpus.py {result['chat_file']}")


if __name__ == "__main__":
    main()
