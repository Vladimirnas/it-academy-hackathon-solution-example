#!/bin/bash
# Сравнивает конфигурации поиска на одном и том же наборе вопросов.
#
# Переключает переменные окружения search-сервиса, перезапускает его и
# прогоняет eval/run_eval.py. Результаты складывает в eval/results/.
#
# Запуск (после docker compose up):  bash eval/ablation.sh
set -uo pipefail

cd "$(dirname "$0")/.."
export OPEN_API_LOGIN="${OPEN_API_LOGIN:-dummy}"
export OPEN_API_PASSWORD="${OPEN_API_PASSWORD:-dummy}"

OUT_DIR="eval/results"
mkdir -p "$OUT_DIR"

run_config() {
  local name="$1"; shift
  echo "=== конфигурация: $name ==="
  # переменные конфигурации передаются как VAR=value перед вызовом
  env "$@" docker compose up -d search >/dev/null 2>&1

  until curl -sf http://localhost:8002/health >/dev/null 2>&1; do sleep 2; done
  sleep 2

  python3 eval/run_eval.py --quiet > "$OUT_DIR/$name.json"
  python3 - "$OUT_DIR/$name.json" "$name" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"  recall={d['recall_avg']:.4f}  ndcg={d['ndcg_avg']:.4f}  score={d['score']:.4f}  ({d['questions']} вопросов)")
PY
  echo
}

run_config "fusion_dbsf" QDRANT_FUSION=DBSF
run_config "fusion_rrf"  QDRANT_FUSION=RRF

echo "=== сводка ==="
python3 - "$OUT_DIR" <<'PY'
import json, sys, pathlib
d = pathlib.Path(sys.argv[1])
rows = []
for f in sorted(d.glob("*.json")):
    r = json.load(open(f))
    rows.append((f.stem, r["recall_avg"], r["ndcg_avg"], r["score"]))
print(f"{'конфигурация':<20} {'recall':>8} {'nDCG':>8} {'score':>8}")
print("-" * 48)
for name, rec, nd, sc in sorted(rows, key=lambda x: -x[2]):
    print(f"{name:<20} {rec:>8.4f} {nd:>8.4f} {sc:>8.4f}")
PY

# вернуть дефолтную конфигурацию
env QDRANT_FUSION=DBSF docker compose up -d search >/dev/null 2>&1
echo
echo "search возвращён в конфигурацию по умолчанию (DBSF)"
