#!/usr/bin/env bash
# Run all milestone-2 benchmarks. Produces bench-out/ with one .txt file per
# scenario you can paste into the report.
#
# Usage: bash setup-vm/run_benchmarks.sh
#
# Assumes:
#  - setup_milestone2.sh has completed.
#  - cwd == repo root.
#  - You have ~50 min of GPU time available.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
# DLAMI ships a pre-built torch venv at /opt/pytorch. Use it directly
# so torch/CUDA/flash-attn ABIs stay matched. (A separate .venv ends up
# with a different cu wheel and triggers the c10::Error ABI mismatch.)
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
    # $1 = log file, $2... = miniengine flags
    local log="$1"; shift
    echo ">>> Starting server: $* >$log"
    python -m miniengine --model "$MODEL" "$@" >"$log" 2>&1 &
    SERVER_PID=$!
    # Wait for /health to come up. Initial weight load + (optional) torch.compile
    # warmup + cuda-graph capture can take 5+ min for the first call.
    local timeout=900
    local elapsed=0
    until curl -sf http://localhost:8000/health >/dev/null 2>&1; do
        if [[ $elapsed -ge $timeout ]]; then
            echo "ERROR: server failed to come up in ${timeout}s. Tail of log:"
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
}start_server "$OUT/server_paged_compile_cudagraph.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 \
    --torch-compile \
    --cuda-graph --cuda-graph-batch-sizes 1,2,4,8,16,32 \
    --cuda-graph-max-blocks 256
run_serving paged_compile_cudagraph
stop_server

# ── 5. Page-size sweep (paged only — page size is what we're varying) ─
start_server "$OUT/server_pagesize256.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256
run_serving pagesize256
stop_server

start_server "$OUT/server_pagesize512.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 512
run_serving pagesize512
stop_server

echo
echo "── ALL BENCHES DONE ─────────────────────────────────────────────"
((pytorch) ) ubuntu@ip-172-31-64-173:~/CS349D-miniengine$ git status
On branch main
Your branch is up to date with 'origin/main'.

Untracked files:
  (use "git add <file>..." to include in what will be committed)
        bench-out/

nothing added to commit but untracked files present (use "git add" to track)
((pytorch) ) ubuntu@ip-172-31-64-173:~/CS349D-miniengine$ bash setup-vm/run_benchmarks.sh
>>> Starting server: --mode paged --mem-fraction-static 0.85 --page-size 256 >bench-out/server_paged.log
  …waiting (5s)
  …waiting (10s)
  server up after 10s (pid 3057)
>>> bench_serving: paged
  Auto-detected model: Qwen/Qwen3-8B

============================================================
  Serving Benchmark
  Server       : http://localhost:8000
  Model        : Qwen/Qwen3-8B
  Requests     : 64 (per concurrency)
  Input len    : 1024
  Output len   : 512
  Randomness   : 0.5
  Concurrencies: [1, 2, 4, 8, 16, 32]
============================================================

  Loading tokenizer...
/opt/pytorch/lib/python3.12/site-packages/torch/cuda/__init__.py:61: FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.
  import pynvml  # type: ignore[import]
  Loading WildChat prompts...
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
  Loaded 320 candidate prompts
  Preparing 64 requests...
  Input lengths:  min=521, max=1013, mean=758
  Output lengths: min=257, max=491, mean=374

  Running concurrency=1 (64 requests)...

[0] 0:bash*                                                                                                                                    "ip-172-31-64-173" 19:51 11-May-26
[7] 0:ssh*                                                                                                                                  "Hivas-MacBook.local" 12:51 11-May-26

run_serving() {
    # $1 = label, $2... = extra bench flags
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
    # $1 = label, $2 = dataset (mmlu | gsm8k)
    local label="$1"
    local dataset="$2"
    echo ">>> bench_accuracy: $label / $dataset"
    python -m benchmark.bench_accuracy \
        --dataset "$dataset" \
        --num-samples "$ACCURACY_SAMPLES" \
        2>&1 | tee "$OUT/accuracy_${label}_${dataset}.txt"
}

trap stop_server EXIT

# NOTE on page size: flash-attn 2.8.x paged kernels require
# `page_block_size` to be a multiple of 256 on Ada-class GPUs (L4 = sm_89).
# Spec values of 32 / 16 / 128 fail; we use 256 for the main paged
# runs and sweep 256 vs 512.

# NOTE on M1 batched (phase 1): commented out in this run because
# (a) M1 throughput is already captured in bench-out/serving_batched.txt
#     from the earlier run, and
# (b) M1 accuracy must be re-run on a fresh server (the previous run's
#     c=32 stress test left the batched server in an OOM loop, so
#     accuracy=0% on the same server is meaningless).
# Re-enable by uncommenting lines below (and remove the matching `: <<'M1_DISABLED'`
# / `M1_DISABLED` heredoc markers).

: <<'M1_DISABLED'
# ── 1. M1 batched (baseline) ──────────────────────────────────────────
start_server "$OUT/server_batched.log" --mode batched
run_serving batched
run_accuracy batched mmlu
stop_server
M1_DISABLED

# ── 2. M2 paged ───────────────────────────────────────────────────────
start_server "$OUT/server_paged.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256
run_serving paged
run_accuracy paged mmlu
stop_server

# ── 3. M2 paged + torch.compile ───────────────────────────────────────
start_server "$OUT/server_paged_compile.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 --torch-compile
run_serving paged_compile
run_accuracy paged_compile mmlu
stop_server

# ── 4. M2 paged + torch.compile + cuda-graph (extra credit) ───────────
start_server "$OUT/server_paged_compile_cudagraph.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256 \
    --torch-compile \
    --cuda-graph --cuda-graph-batch-sizes 1,2,4,8,16,32 \
    --cuda-graph-max-blocks 256
run_serving paged_compile_cudagraph
stop_server

# ── 5. Page-size sweep (paged only — page size is what we're varying) ─
start_server "$OUT/server_pagesize256.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 256
run_serving pagesize256
stop_server

start_server "$OUT/server_pagesize512.log" \
    --mode paged --mem-fraction-static 0.85 --page-size 512
run_serving pagesize512
stop_server

echo
echo "── ALL BENCHES DONE ─────────────────────────────────────────────"
echo "Outputs in: $OUT/"
ls -la "$OUT/"
