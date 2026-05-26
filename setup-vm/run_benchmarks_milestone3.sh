#!/usr/bin/env bash
# Milestone-3 bench harness. Each subcommand calls one or more of
# bench_cache.py / bench_serving.py / bench_accuracy.py with the right
# args and writes a single, idempotent .txt under bench-out/.
#
# Server-state expectations are documented per subcommand. The script
# itself never starts/stops the server — that's a manual step in the
# server shell, because Ctrl-C is the only safe way to swap server
# flags.
#
# Usage:
#   bash run_benchmarks_milestone3.sh oom_no_chunk
#   bash run_benchmarks_milestone3.sh oom_chunked
#   bash run_benchmarks_milestone3.sh mmlu_no_chunk
#   bash run_benchmarks_milestone3.sh mmlu_chunked
#   bash run_benchmarks_milestone3.sh serving_chunk_off
#   bash run_benchmarks_milestone3.sh serving_chunk_on
#   bash run_benchmarks_milestone3.sh serving_cache_off
#   bash run_benchmarks_milestone3.sh serving_cache_on
#   bash run_benchmarks_milestone3.sh retract_off
#   bash run_benchmarks_milestone3.sh retract_on

set -u
BASE_URL="${BASE_URL:-http://localhost:8000}"
OUT="bench-out"
mkdir -p "$OUT"

# Wait until the server's /health responds before kicking off a bench.
# Lets the caller pipe `bash run_benchmarks_milestone3.sh foo` right
# after restarting the server in another tab.
wait_for_server() {
  local i=0
  while ! curl -s --max-time 1 "$BASE_URL/health" >/dev/null 2>&1; do
    i=$((i+1))
    if [ $i -gt 180 ]; then
      echo "ERROR: server at $BASE_URL never responded" >&2
      exit 1
    fi
    sleep 1
  done
}

PHASE="${1:-help}"

case "$PHASE" in

  # ---------- Item #1: chunked prefill OOM avoidance ---------------
  # Expect: server in --mode paged --disable-radix-cache --page-size 256
  # First run with --prefill-chunk-size 0 (should OOM or crash on
  # activations under the long-prompt heavy workload). Second run with
  # the server restarted at --prefill-chunk-size 512 (should complete).
  oom_no_chunk)
    wait_for_server
    # Long prompts at high concurrency -> packed varlen prefill activation
    # blow-up at chunk=0. 16k input × conc 8 = 128k Q-tokens in one fwd.
    python -m benchmark.bench_serving \
      --base-url "$BASE_URL" \
      --num-requests 32 \
      --concurrencies 8 \
      --input-len 16384 \
      --output-len 64 \
      | tee "$OUT/m3_oom_chunk0.txt"
    ;;

  oom_chunked)
    wait_for_server
    python -m benchmark.bench_serving \
      --base-url "$BASE_URL" \
      --num-requests 32 \
      --concurrencies 8 \
      --input-len 16384 \
      --output-len 64 \
      | tee "$OUT/m3_oom_chunk512.txt"
    ;;

  # ---------- Item #2: MMLU accuracy parity ------------------------
  # Both runs server in --mode paged --disable-radix-cache --page-size 256
  # First with --prefill-chunk-size 0, second with --prefill-chunk-size 512.
  mmlu_no_chunk)
    wait_for_server
    python -m benchmark.bench_accuracy \
      --base-url "$BASE_URL" \
      --dataset mmlu \
      --num-samples 200 \
      --concurrency 8 \
      | tee "$OUT/m3_mmlu_chunk0.txt"
    ;;

  mmlu_chunked)
    wait_for_server
    python -m benchmark.bench_accuracy \
      --base-url "$BASE_URL" \
      --dataset mmlu \
      --num-samples 200 \
      --concurrency 8 \
      | tee "$OUT/m3_mmlu_chunk512.txt"
    ;;

  # ---------- Item #3: serving no-regression chunked vs unchunked --
  # Server in --mode paged --disable-radix-cache (no cache effect),
  # first with chunk=0, then with chunk=512. Pick input-len so chunking
  # actually fires: 4096 input / 512 chunk = 8 chunks per request.
  serving_chunk_off)
    wait_for_server
    python -m benchmark.bench_serving \
      --base-url "$BASE_URL" \
      --num-requests 64 \
      --concurrencies 1,4,16 \
      --input-len 4096 \
      --output-len 256 \
      | tee "$OUT/m3_serving_chunk0.txt"
    ;;

  serving_chunk_on)
    wait_for_server
    python -m benchmark.bench_serving \
      --base-url "$BASE_URL" \
      --num-requests 64 \
      --concurrencies 1,4,16 \
      --input-len 4096 \
      --output-len 256 \
      | tee "$OUT/m3_serving_chunk512.txt"
    ;;

  # ---------- Item #7: serving no-regression cache off vs on -------
  # First with --disable-radix-cache, then default (cache on). Default
  # WildChat-ish input/output lengths so we measure cache OVERHEAD, not
  # cache wins.
  serving_cache_off)
    wait_for_server
    python -m benchmark.bench_serving \
      --base-url "$BASE_URL" \
      --num-requests 64 \
      --concurrencies 1,4,16 \
      --input-len 1024 \
      --output-len 512 \
      | tee "$OUT/m3_serving_cache_off.txt"
    ;;

  serving_cache_on)
    wait_for_server
    python -m benchmark.bench_serving \
      --base-url "$BASE_URL" \
      --num-requests 64 \
      --concurrencies 1,4,16 \
      --input-len 1024 \
      --output-len 512 \
      | tee "$OUT/m3_serving_cache_on.txt"
    ;;

  # ---------- Items #8 / #9: retraction bonus ----------------------
  # Same overrun workload twice. retract_off expects server WITHOUT
  # --enable-retraction; some requests should fail with KV OOM. retract_on
  # expects server WITH --enable-retraction; all should complete.
  retract_off|retract_on)
    wait_for_server
    suffix=$([ "$PHASE" = "retract_off" ] && echo off || echo on)
    python -m benchmark.bench_serving \
      --base-url "$BASE_URL" \
      --num-requests 48 \
      --concurrencies 24 \
      --input-len 2048 \
      --output-len 1024 \
      | tee "$OUT/m3_retract_${suffix}.txt"
    ;;

  *)
    cat <<EOF
usage: bash $0 <phase>

phases (require the matching server state in another tab):

  oom_no_chunk        --mode paged --disable-radix-cache (chunk=0)
  oom_chunked         --mode paged --disable-radix-cache --prefill-chunk-size 512

  mmlu_no_chunk       --mode paged --disable-radix-cache (chunk=0)
  mmlu_chunked        --mode paged --disable-radix-cache --prefill-chunk-size 512

  serving_chunk_off   --mode paged --disable-radix-cache (chunk=0)
  serving_chunk_on    --mode paged --disable-radix-cache --prefill-chunk-size 512

  serving_cache_off   --mode paged --disable-radix-cache
  serving_cache_on    --mode paged                         (default)

  retract_off         --mode paged                         (cache on, retract off)
  retract_on          --mode paged --enable-retraction

writes to bench-out/m3_<phase>.txt.
EOF
    exit 1
    ;;
esac
