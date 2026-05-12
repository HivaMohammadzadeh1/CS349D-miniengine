#!/usr/bin/env bash
# Supplement run: fills in the deliverables the main run_benchmarks.sh
# didn't produce. Three phases:
#   1. cuda-graph EC retry (now that the RoPE .item() bug is fixed)
#   2. Page-size 512 sweep (page-size 256 is already in serving_paged.txt)
#   3. M1 batched accuracy on a FRESH server (the first attempt got 0%
#      because the M1 server was OOM-stuck after c=32 stress)
#
# Run inside tmux so SSH drops don't kill it:
#   tmux new -s supp
#   bash setup-vm/run_benchmarks_supplement.sh
#   Ctrl-b d  (detach)
#   tmux attach -t supp  (reattach later)
#
# Total wall time: ~70 min if cuda-graph capture succeeds on first try.

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

# ── Helpers ───────────────────────────────────────────────────────────

start_server() {
    local log="$1"; shift
    echo ">>> Starting server: $* >$log"
    python -m miniengine --model "$MODEL" "$@" >"$log" 2>&1 &
    SERVER_PID=$!
    local timeout=900
    local elapsed=0
    until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo "ERROR: server process died. Tail of log:"
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

# Reap any zombie miniengine processes hogging GPU before we start.
echo "── Pre-flight: clearing stale GPU processes ─────────────────────"
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9 2>/dev/null || true
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader

trap stop_server EXIT

# ── 1. cuda-graph EC retry ───────────────────────────────────────────
# Depends on the RoPE .item() fix on main (commit 4c89b2b or later).
echo
echo "════════════════════════════════════════════════════════════════"
echo "  PHASE 1/3 — paged + torch.compile + cuda-graph (EC retry)"
echo "════════════════════════════════════════════════════════════════"
if start_server "$OUT/server_paged_compile_cudagraph.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 \
    --torch-compile \
    --cuda-graph --cuda-graph-batch-sizes 1,2,4,8,16,32 \
    --cuda-graph-max-blocks 256; then
    run_serving paged_compile_cudagraph
    stop_server
    echo "✅ cuda-graph EC phase complete"
else
    echo "⚠️  cuda-graph capture still failing — skipping EC throughput."
    echo "    See $OUT/server_paged_compile_cudagraph.log for details."
    stop_server
fi

# ── 2. Page-size 512 sweep ───────────────────────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  PHASE 2/3 — page-size 512 sweep"
echo "════════════════════════════════════════════════════════════════"
start_server "$OUT/server_pagesize512.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 512
run_serving pagesize512
stop_server
echo "✅ page-size 512 phase complete"

# ── 3. M1 batched accuracy on a FRESH server ─────────────────────────
echo
echo "════════════════════════════════════════════════════════════════"
echo "  PHASE 3/3 — M1 batched accuracy (fresh server)"
echo "════════════════════════════════════════════════════════════════"
start_server "$OUT/server_batched_accuracy.log" --mode batched
run_accuracy batched mmlu
stop_server
echo "✅ M1 batched accuracy phase complete"

echo
echo "── ALL SUPPLEMENT BENCHES DONE ──────────────────────────────────"
echo "Outputs in: $OUT/"
ls -la "$OUT/" | grep -E "(serving_paged_compile_cudagraph|serving_pagesize512|accuracy_batched_mmlu)"
