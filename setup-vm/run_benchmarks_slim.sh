#!/usr/bin/env bash
# Cut-down milestone-2 benchmarks: 3 core modes, no accuracy, no sweep.
# Designed for /opt/pytorch env (DLAMI), not .venv.

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source /opt/pytorch/bin/activate

OUT=bench-out
mkdir -p "$OUT"

MODEL=Qwen/Qwen3-8B
NUM_REQUESTS=64
INPUT_LEN=1024
OUTPUT_LEN=512
CONCURRENCIES="1,2,4,8,16,32"

start_server() {
    local log="$1"; shift
    echo ">>> Starting server: $* >$log"
    python -m miniengine --model "$MODEL" "$@" >"$log" 2>&1 &
    SERVER_PID=$!
    local timeout=900
    local elapsed=0
    until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        if [[ $elapsed -ge $timeout ]]; then
            echo "ERROR: server failed in ${timeout}s. Tail:"
            tail -40 "$log"
            kill $SERVER_PID 2>/dev/null || true
            exit 1
        fi
        sleep 5
        elapsed=$((elapsed + 5))
        echo "  …waiting (${elapsed}s)"
    done
    echo "  server up after ${elapsed}s (pid $SERVER_PID)"
}

stop_server() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID"
        wait "$SERVER_PID" 2>/dev/null || true
    fi
    SERVER_PID=
    sleep 3
}

run_serving() {
    local label="$1"; shift
    echo ">>> bench_serving: $label"
    python -m benchmark.bench_serving \
        --num-requests "$NUM_REQUESTS" \
        --input-len "$INPUT_LEN" \
        --output-len "$OUTPUT_LEN" \
        --concurrencies "$CONCURRENCIES" \
        "$@" 2>&1 | tee "$OUT/serving_${label}.txt"
}

trap stop_server EXIT

# 1. M1 batched
start_server "$OUT/server_batched.log" --mode batched
run_serving batched
stop_server

# 2. M2 paged  (page-size must be a multiple of 256 for current flash-attn — see Ed #47)
start_server "$OUT/server_paged.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256
run_serving paged
stop_server

# 3. M2 paged + torch.compile
start_server "$OUT/server_paged_compile.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 --torch-compile
run_serving paged_compile
stop_server

echo
echo "── DONE ─────────────────────────────────────────────"
ls -la "$OUT/"
