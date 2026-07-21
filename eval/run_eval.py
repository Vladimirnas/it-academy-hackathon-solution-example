"""Считает Recall@50 / nDCG@50 / score по формуле на своём наборе
вопросов (eval/questions.json), обращаясь к живому search-сервису.

score = recall_avg * 0.8 + ndcg_avg * 0.2  

ВАЖНО: корпус (data/Go Nova.json) даёт всего ~5 чанков - на таком размере
retrieval-лимиты (RETRIEVE_K=100, TIER1_LIMIT=35, MAX_MESSAGE_IDS=50) с
большим запасом перекрывают весь корпус, поэтому Recall@50 предсказуемо
насыщается около 1.0 почти независимо от качества поиска. Единственная
метрика, которая тут реально что-то показывает - nDCG@50 (чувствительна
к порядку выдачи, то есть к работе reranker'а).
"""

import json
import math
import sys
import urllib.request
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


QUESTIONS_FILE = Path(__file__).parent / "questions.json"
SEARCH_URL = "http://localhost:8002/search"
K = 50


def http_post_json(url: str, payload: dict, timeout: float = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def recall_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    top_k = set(predicted[:k])
    return len(top_k & gold) / len(gold)


def ndcg_at_k(predicted: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, mid in enumerate(predicted[:k], start=1)
        if mid in gold
    )
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def main() -> None:
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    rows = []
    for case in cases:
        question = case["question"]
        gold = set(case["gold_message_ids"])

        response = http_post_json(SEARCH_URL, {"question": question})
        results = response.get("results") or []
        predicted = results[0]["message_ids"] if results else []

        recall = recall_at_k(predicted, gold, K)
        ndcg = ndcg_at_k(predicted, gold, K)
        rows.append((question["text"], len(gold), len(predicted), recall, ndcg))

    recall_avg = sum(r[3] for r in rows) / len(rows)
    ndcg_avg = sum(r[4] for r in rows) / len(rows)
    score = recall_avg * 0.8 + ndcg_avg * 0.2

    if "--quiet" in sys.argv:
        print(json.dumps({
            "recall_avg": round(recall_avg, 4),
            "ndcg_avg": round(ndcg_avg, 4),
            "score": round(score, 4),
            "questions": len(rows),
            "per_question": [
                {"question": t, "recall": round(r, 4), "ndcg": round(n, 4)}
                for t, _, _, r, n in rows
            ],
        }, ensure_ascii=False))
        return

    print(f"{'Вопрос':<70} {'|gold|':>7} {'|pred|':>7} {'Recall@50':>10} {'nDCG@50':>10}")
    print("-" * 108)
    for text, n_gold, n_pred, recall, ndcg in rows:
        short = text if len(text) <= 68 else text[:65] + "..."
        print(f"{short:<70} {n_gold:>7} {n_pred:>7} {recall:>10.3f} {ndcg:>10.3f}")

    print("-" * 108)
    print(f"recall_avg = {recall_avg:.4f}")
    print(f"ndcg_avg   = {ndcg_avg:.4f}")
    print(f"score      = {score:.4f}   (recall_avg*0.8 + ndcg_avg*0.2)")


if __name__ == "__main__":
    main()
