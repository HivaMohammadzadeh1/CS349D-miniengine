#!/usr/bin/env bash
# EC-only retry: runs the cuda-graph extra-credit phase by itself.
# Use this after the main supplement script has finished phases 2 and 3,
# once the cuda-graph capture bug has been patched on main.
#
# Requires the following fixes:
#   - 4c89b2b: RoPE .item() in capture
#   - <next>:  per-bucket compile warmup before capture
#
# Usage on the GPU instance:
#   git pull origin main
#   tmux new -s ec
#   bash setup-vm/run_benchmarks_ec_only.sh
#   Ctrl-b d
#
# Wall time: ~30 min (capture warmup + 6-concurrency sweep).

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
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server died. Tail of log:"
            tail -60 "$log"
            return 1
        fi
        if [[ $elapsed -ge $timeout ]]; then
            echo "ERROR: server failed in ${timeout}s. Tail of log:"
            tail -60 "$log"
            kill "$SERVER_PID" 2>/dev/null || true
            return 1
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

# Pre-flight
echo "── Pre-flight: clearing stale GPU processes ─────────────────────"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

trap stop_server EXIT

echo
echo "════════════════════════════════════════════════════════════════"
echo "  cuda-graph EC retry"
echo "════════════════════════════════════════════════════════════════"
start_server "$OUT/server_paged_compile_cudagraph.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 \
    --torch-compile \
    --cuda-graph --cuda-graph-batch-sizes 1,2,4,8,16,32 \
    --cuda-graph-max-blocks 256
run_serving paged_compile_cudagraph
stop_server

echo
echo "── EC PHASE DONE ────────────────────────────────────────────────"
ls -la "$OUT/serving_paged_compile_cudagraph.txt" "$OUT/server_paged_compile_cudagraph.log"
