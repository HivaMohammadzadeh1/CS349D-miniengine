"""Round-robin multi-turn re-access bench (milestone 4 HiCache perf-win demo).

The shipped bench_cache.multiturn worker pool runs all turns of one session
to completion before pulling the next session, so a session's prior-turn
prefix is always the freshest cached entry when its next turn looks for it.
LRU eviction (whether drop or demote) only hits prefixes of finished
sessions, which the workload never re-accesses -- so HiCache's
avoided-re-prefill savings never materialize.

This bench fixes that. Sessions advance turn-by-turn IN LOCKSTEP across the
whole pool: every session does turn 0 first (in some order), then every
session does turn 1, and so on. Between session A's turn k and turn k+1,
*every other session* does a turn first; their cache touches refresh the
LRU on B..Z and push A's prefix toward eviction. With enough sessions, by
the time A's turn k+1 starts, A's turn k prefix may be the LRU victim --
exactly the re-access pattern HiCache rescues from a re-prefill.

Usage:
    python -m benchmark.bench_reaccess \
        --base-url http://localhost:8000 \
        --num-sessions 96 --turns 6 --max-tokens 192 --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import time

import aiohttp


SYSTEM_TEMPLATE = (
    "You are a helpful encyclopedic assistant. Session {sid}. Be concise. "
    "Answer the user's question directly and stop. "
) * 32   # ~700-900 Qwen3 tokens => each session gets ~3 pages of cached
         # prefix from turn 0 onward, so cache pressure scales linearly
         # with session count and HiCache's avoided-re-prefill saving is
         # large per hit.


USER_TURNS = [
    "Tell me a fun fact about the moon.",
    "Roughly how big is it?",
    "What is it made of?",
    "Has anyone landed on it?",
    "What is the dark side like?",
    "How long does it take to orbit Earth?",
    "Why do we see phases?",
    "What is the tidal effect?",
    "Could humans live there?",
    "Does it have a magnetic field?",
]


async def stream_chat(session, base_url, messages, max_tokens):
    """Streaming completion. Returns (text, usage, t_start, t_first, t_end)."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {
        "model": "qwen3-8b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    t_start = time.perf_counter()
    t_first = None
    parts = []
    usage = {}
    async with session.post(url, json=body) as resp:
        async for raw in resp.content:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            ch = (obj.get("choices") or [{}])[0]
            delta = ch.get("delta", {}).get("content", "")
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()
                parts.append(delta)
            u = obj.get("usage")
            if u:
                usage = u
    t_end = time.perf_counter()
    return "".join(parts), usage, t_start, (t_first or t_end), t_end


async def run(args):
    rng = random.Random(42)

    # Per-session conversation state.
    histories: list[list[dict]] = [
        [{"role": "system", "content": SYSTEM_TEMPLATE.format(sid=sid)}]
        for sid in range(args.num_sessions)
    ]

    # Round-robin order: all turn 0 (sessions in random shuffled order, fixed
    # seed), then all turn 1, ... Shuffle each turn so the access pattern
    # isn't trivially identical across turns -- this makes the LRU's job
    # harder, which is what we want.
    schedule: list[tuple[int, int]] = []
    for turn in range(args.turns):
        order = list(range(args.num_sessions))
        rng.shuffle(order)
        for sid in order:
            schedule.append((turn, sid))

    print(f"Schedule: {len(schedule)} requests")
    print(f"  Sessions    : {args.num_sessions}")
    print(f"  Turns       : {args.turns}")
    print(f"  Max tokens  : {args.max_tokens}")
    print(f"  Concurrency : {args.concurrency}")
    print(f"  Order       : round-robin (all turn k before any turn k+1)")
    print()

    # Per-turn collected samples (ttft, latency, hit_tokens, prompt_tokens).
    per_turn: dict[int, dict] = {
        t: {"ttft": [], "latency": [], "hit_tok": 0, "prompt_tok": 0, "n": 0}
        for t in range(args.turns)
    }

    sem = asyncio.Semaphore(args.concurrency)

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600),
    ) as session:
        # Inside one turn, fire all sessions' requests concurrently
        # (bounded by --concurrency). After all of turn k complete, move
        # to turn k+1. This is what "round-robin" means here -- a hard
        # synchronization barrier between turns.
        for turn in range(args.turns):
            turn_t0 = time.perf_counter()
            tasks = []
            for sid in [s for (t, s) in schedule if t == turn]:
                tasks.append(asyncio.create_task(
                    one_request(session, args, turn, sid, histories,
                                per_turn, sem, rng)
                ))
            # return_exceptions so a single transient ConnectionResetError
            # doesn't kill the whole barrier -- log it, drop the sample.
            results = await asyncio.gather(*tasks, return_exceptions=True)
            failures = [r for r in results if isinstance(r, Exception)]
            if failures:
                print(f"    [warn] {len(failures)} request(s) failed: "
                      f"{type(failures[0]).__name__}: {failures[0]}")
            print(
                f"  turn {turn} done in {time.perf_counter() - turn_t0:.1f}s "
                f"-- hit_rate={100 * per_turn[turn]['hit_tok'] / max(1, per_turn[turn]['prompt_tok']):.1f}%"
            )

    # ── Summary
    print()
    print("=" * 64)
    print("  Round-robin multi-turn re-access bench")
    print("=" * 64)
    print(f"{'turn':>4}  {'N':>4}  {'prompt_tok':>10}  {'hit_tok':>9}  "
          f"{'hit_rate':>8}  {'TTFT_p50':>9}  {'TTFT_p99':>9}  {'lat_p50':>9}")
    total_prompt = 0
    total_hit = 0
    all_ttft = []
    all_lat = []
    for t in range(args.turns):
        s = per_turn[t]
        if not s["ttft"]:
            continue
        ttft_p50 = statistics.median(s["ttft"])
        ttft_p99 = sorted(s["ttft"])[int(0.99 * (len(s["ttft"]) - 1))]
        lat_p50 = statistics.median(s["latency"])
        hr = 100 * s["hit_tok"] / max(1, s["prompt_tok"])
        print(f"{t:>4}  {s['n']:>4}  {s['prompt_tok']:>10}  "
              f"{s['hit_tok']:>9}  {hr:>7.1f}%  "
              f"{ttft_p50*1000:>7.0f}ms  {ttft_p99*1000:>7.0f}ms  "
              f"{lat_p50*1000:>7.0f}ms")
        total_prompt += s["prompt_tok"]
        total_hit += s["hit_tok"]
        all_ttft.extend(s["ttft"])
        all_lat.extend(s["latency"])

    if all_ttft:
        print()
        print(f"  Overall hit rate    : {100 * total_hit / max(1, total_prompt):.1f}%")
        print(f"  Overall TTFT p50/p99: {statistics.median(all_ttft)*1000:.0f} / "
              f"{sorted(all_ttft)[int(0.99 * (len(all_ttft) - 1))]*1000:.0f} ms")
        print(f"  Overall latency p50 : {statistics.median(all_lat)*1000:.0f} ms")


async def one_request(session, args, turn, sid, histories, per_turn, sem, rng):
    user_msg = USER_TURNS[(rng.randrange(len(USER_TURNS)) + turn) % len(USER_TURNS)]
    history = histories[sid]
    history.append({"role": "user", "content": user_msg})

    async with sem:
        text, usage, t_start, t_first, t_end = await stream_chat(
            session, args.base_url, history, args.max_tokens,
        )

    history.append({"role": "assistant", "content": text})
    per_turn[turn]["ttft"].append(t_first - t_start)
    per_turn[turn]["latency"].append(t_end - t_start)
    per_turn[turn]["prompt_tok"] += usage.get("prompt_tokens", 0)
    per_turn[turn]["hit_tok"] += usage.get("cache_hit_tokens", 0)
    per_turn[turn]["n"] += 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://localhost:8000")
    p.add_argument("--num-sessions", type=int, default=96)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--max-tokens", type=int, default=192)
    p.add_argument("--concurrency", type=int, default=32)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
