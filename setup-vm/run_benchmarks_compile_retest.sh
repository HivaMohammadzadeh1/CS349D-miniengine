#!/usr/bin/env bash
# Retest of paged+compile and paged+compile+cuda-graph after the
# decode-only-MLP compile fix (commit bf79dfb).
#
# Expected outcomes after the fix:
#   * paged+compile @ c=8 ≥ 100 tok/s (≥ +10% over paged's 91 — required §3.3.3)
#   * paged+compile+cuda-graph @ c=8 ≥ 110 tok/s (further +10% from graphs)
#   * c=16 and c=32 no longer break for paged+compile (no recompile thrashing)
#
# Usage on the GPU instance:
#   git pull origin main
#   tmux new -s retest
#   bash setup-vm/run_benchmarks_compile_retest.sh
#   Ctrl-b d
#
# Wall time: ~60–80 min.

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
ACCURACY_SAMPLES=200

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

run_accuracy() {
    local label="$1"
    local dataset="$2"
    echo ">>> bench_accuracy: $label / $dataset"
    python -m benchmark.bench_accuracy \
        --dataset "$dataset" \
        --num-samples "$ACCURACY_SAMPLES" \
        2>&1 | tee "$OUT/accuracy_${label}_${dataset}.txt"
}

echo "── Pre-flight: clearing stale GPU processes ─────────────────────"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

trap stop_server EXIT

# ── 1. paged + torch.compile (re-bench) ──────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  PHASE 1/2 — paged + torch.compile RETEST (decode-only compile)"
echo "════════════════════════════════════════════════════════════════"
start_server "$OUT/server_paged_compile.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 --torch-compile
run_serving paged_compile
run_accuracy paged_compile mmlu
stop_server
echo "✅ paged + torch.compile retest complete"

# ── 2. paged + torch.compile + cuda-graph (re-bench) ─────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  PHASE 2/2 — paged + torch.compile + cuda-graph RETEST (EC)"
echo "════════════════════════════════════════════════════════════════"
start_server "$OUT/server_paged_compile_cudagraph.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 \
    --torch-compile \
    --cuda-graph --cuda-graph-batch-sizes 1,2,4,8,16,32 \
    --cuda-graph-max-blocks 256
run_serving paged_compile_cudagraph
stop_server
echo "✅ paged + torch.compile + cuda-graph retest complete"

echo
echo "── ALL COMPILE RETESTS DONE ─────────────────────────────────────"
echo "Compare the new summary tables against the prior numbers (the"
echo "old serving_paged_compile.txt before this commit had:"
echo "    c=8 throughput 78 tok/s, c=16 broken at 50/64 ok)."
echo
ls -la "$OUT/" | grep -E "(serving_paged_compile|accuracy_paged_compile)"
