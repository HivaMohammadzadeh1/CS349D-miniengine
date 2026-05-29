#!/usr/bin/env bash
# Run the HiCache cliff/restore demo (milestone 4 Track 1, full-credit bar).
#
# Two passes against the same bench_cache --workload multiturn workload:
#   pass A: --cpu-cache-size-gb 0  (HiCache off = milestone-3 baseline)
#   pass B: --cpu-cache-size-gb $CPU_GB  (HiCache on)
#
# Expected outcome:
#   A: per-turn hit_rate starts high (turn 1 ≥70%) and collapses by the
#      last turn (<20%) once the working set exceeds the GPU pool.
#   B: per-turn hit_rate stays high across all turns — the demoted prefixes
#      are still in the tree (CPU-tier) and get promoted back on hit.
#
# Usage from the repo root:
#   bash scripts/hicache_multiturn_demo.sh
# Override knobs via env:
#   CPU_GB=40 NUM_SESSIONS=32 TURNS=5 MAX_TOKENS=256 bash scripts/hicache_multiturn_demo.sh

set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-Qwen/Qwen3-8B}
CPU_GB=${CPU_GB:-40}
NUM_SESSIONS=${NUM_SESSIONS:-32}
TURNS=${TURNS:-5}
MAX_TOKENS=${MAX_TOKENS:-256}
CONCURRENCY=${CONCURRENCY:-1}
PORT=${PORT:-8000}
MEM_FRAC=${MEM_FRAC:-0.85}
CHUNK=${CHUNK:-512}
OUTDIR=${OUTDIR:-bench-out}
PY=${PY:-.venv/bin/python}

mkdir -p "$OUTDIR"

run_pass () {
    local label="$1"
    local cpu_gb_arg="$2"
    local outfile="$OUTDIR/hicache_${label}.txt"
    local statsfile="$OUTDIR/hicache_${label}.cache_stats.json"
    local serverlog="$OUTDIR/hicache_${label}.server.log"

    echo "=========================================================="
    echo "[$label] launching server  cpu-cache-size-gb=$cpu_gb_arg"
    echo "=========================================================="

    pkill -f "python -m miniengine" 2>/dev/null; sleep 2

    $PY -m miniengine \
        --model "$MODEL" --mode paged \
        --mem-fraction-static "$MEM_FRAC" --page-size 32 \
        --prefill-chunk-size "$CHUNK" \
        --cpu-cache-size-gb "$cpu_gb_arg" \
        --port "$PORT" \
        > "$serverlog" 2>&1 &
    local pid=$!

    # Poll for readiness (HTTP responding).
    for i in $(seq 1 300); do
        code=$(curl -s -m 1 "http://localhost:$PORT/health" -o /dev/null -w "%{http_code}" 2>/dev/null || true)
        if [ "$code" = "200" ]; then echo "[$label] ready after ${i}s"; break; fi
        if ! kill -0 $pid 2>/dev/null; then
            echo "[$label] server crashed during boot"; tail -20 "$serverlog"; return 1
        fi
        sleep 1
    done

    echo "[$label] HiCache + KV pool init from server log:"
    grep -E "HiCache|KV pool|Radix prefix" "$serverlog" | sed 's/^/    /'

    echo "[$label] running bench_cache multiturn  sessions=$NUM_SESSIONS  turns=$TURNS  max-tokens=$MAX_TOKENS  conc=$CONCURRENCY"
    $PY -m benchmark.bench_cache \
        --workload multiturn \
        --num-sessions "$NUM_SESSIONS" \
        --turns-per-session "$TURNS" \
        --max-tokens "$MAX_TOKENS" \
        --concurrency "$CONCURRENCY" \
        --base-url "http://localhost:$PORT" \
        2>&1 | tee "$outfile"

    echo "[$label] /cache_stats snapshot:"
    curl -s "http://localhost:$PORT/cache_stats" | tee "$statsfile" | $PY -m json.tool || echo "(stats parse failed)"

    kill $pid 2>/dev/null; sleep 1
    echo
}

# Pass A: baseline (HiCache off, GPU-only radix). Same binary, flag = 0.
run_pass "off" 0

# Pass B: HiCache on.
run_pass "on" "$CPU_GB"

echo "=========================================================="
echo "Done. Output files in $OUTDIR/:"
ls -la "$OUTDIR"/hicache_off.* "$OUTDIR"/hicache_on.* 2>/dev/null
