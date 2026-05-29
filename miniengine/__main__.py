"""
CLI entry point — launch the MiniEngine server.

Usage:
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507
    python -m miniengine --model Qwen/Qwen3-4B-Instruct-2507 --port 8080 --dtype bfloat16
"""

from __future__ import annotations

import argparse
import logging

import torch
import uvicorn

from miniengine.engine import Engine
from miniengine.scheduler import Scheduler
from miniengine import server as srv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="miniengine",
        description="Minimal LLM serving engine",
    )
    p.add_argument(
        "--model", type=str, required=True, help="HuggingFace model id or local path"
    )
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16", "float32"],
    )
    p.add_argument("--device", type=str, default="cuda", help="Device to load model on")
    p.add_argument(
        "--max-running",
        type=int,
        default=16,
        help="Max concurrent requests in the scheduler",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="batched",
        choices=["baseline", "batched", "paged"],
        help="Scheduling mode: baseline (one request at a time), "
        "batched (iteration-level batching, milestone 1), or "
        "paged (paged KV pool + flash-attn paged attention, milestone 2)",
    )
    # ── Milestone 2 flags (only meaningful when --mode paged) ────────────
    p.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.85,
        help="Fraction of total GPU memory to pre-allocate for static "
        "tensors (model weights + KV pool). Pool capacity is derived from "
        "what's left after weights.",
    )
    p.add_argument(
        "--page-size",
        type=int,
        default=32,
        help="Tokens per KV page. Smaller = less tail-fragmentation, "
        "bigger page tables; larger = the opposite.",
    )
    p.add_argument(
        "--torch-compile",
        action="store_true",
        help="torch.compile the per-layer MLP (stable shapes, "
        "dynamic batch). Targets ~10%% speedup on decode.",
    )
    p.add_argument(
        "--cuda-graph",
        action="store_true",
        help="(Extra credit) Capture paged decode as CUDA graphs at the "
        "configured bucket batch sizes. Implies --torch-compile in "
        "non-cudagraph mode (compiled kernels are wrapped by the manual graph).",
    )
    p.add_argument(
        "--cuda-graph-batch-sizes",
        type=str,
        default="1,2,4,8,16,32",
        help="Comma-separated bucket batch sizes for --cuda-graph.",
    )
    p.add_argument(
        "--cuda-graph-max-blocks",
        type=int,
        default=256,
        help="Max page-table length per request supported by the captured "
        "graphs. Bound on prompt+output length / page-size.",
    )
    # ── Milestone 3 flags ────────────────────────────────────────────────
    p.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=0,
        help="Per-step prefill q-token budget. 0 disables chunking "
        "(milestone-2 single-shot path). When > 0, prefill processes one "
        "chunk per scheduler step; decode of other running requests "
        "continues in parallel.",
    )
    p.add_argument(
        "--disable-radix-cache",
        action="store_true",
        help="Disable the radix prefix cache. Cache is on by default in "
        "--mode paged (milestone 3 Part B).",
    )
    p.add_argument(
        "--enable-retraction",
        action="store_true",
        help="Enable decode-time retraction (milestone 3 bonus). When the "
        "KV pool runs out mid-decode, the scheduler evicts a victim "
        "(youngest by arrival time) back to the waiting queue.",
    )
    # ── Milestone 4 (HiCache, Track 1) ───────────────────────────────────
    p.add_argument(
        "--cpu-cache-size-gb",
        type=float,
        default=0.0,
        help="Size of the CPU KV tier in GiB (HiCache, milestone 4). 0 "
        "disables HiCache and the radix cache stays GPU-only (byte-"
        "identical to milestone 3). Requires --mode paged and the radix "
        "cache enabled.",
    )
    p.add_argument(
        "--hicache-overlap",
        action="store_true",
        help="Use a dedicated CUDA stream + pinned host memory for async "
        "D2H/H2D demote/promote. Only meaningful when "
        "--cpu-cache-size-gb > 0. Default off — implement first as a "
        "blocking copy, flip this on for the perf-bonus runs.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("miniengine")

    dtype = getattr(torch, args.dtype)
    logger.info(
        "Initializing engine  model=%s  dtype=%s  mode=%s",
        args.model,
        args.dtype,
        args.mode,
    )

    cuda_graph_batch_sizes = [
        int(s.strip()) for s in args.cuda_graph_batch_sizes.split(",") if s.strip()
    ]
    if args.cuda_graph and args.mode != "paged":
        logger.warning(
            "--cuda-graph only applies to --mode paged; ignoring."
        )
    if args.cuda_graph and not args.torch_compile:
        logger.info(
            "--cuda-graph implicitly enables --torch-compile (compiled "
            "kernels are captured inside the manual graph)."
        )
    if args.mode != "paged":
        if args.prefill_chunk_size > 0:
            logger.warning(
                "--prefill-chunk-size only applies to --mode paged; ignoring."
            )
        if args.disable_radix_cache:
            logger.warning(
                "--disable-radix-cache only applies to --mode paged; ignoring."
            )
        if args.enable_retraction:
            logger.warning(
                "--enable-retraction only applies to --mode paged; ignoring."
            )
        if args.cpu_cache_size_gb > 0:
            raise SystemExit(
                "--cpu-cache-size-gb (HiCache) requires --mode paged."
            )

    # HiCache fail-fast: overlap is meaningless without a CPU pool, and the
    # CPU pool needs the radix cache to attach to.
    if args.hicache_overlap and args.cpu_cache_size_gb <= 0:
        raise SystemExit(
            "--hicache-overlap requires --cpu-cache-size-gb > 0."
        )
    if args.cpu_cache_size_gb > 0 and args.disable_radix_cache:
        raise SystemExit(
            "--cpu-cache-size-gb requires the radix cache; remove "
            "--disable-radix-cache."
        )

    engine = Engine(
        model_path=args.model,
        dtype=dtype,
        device=args.device,
        mode=args.mode,
        page_size=args.page_size,
        mem_fraction_static=args.mem_fraction_static,
        torch_compile=args.torch_compile or (args.cuda_graph and args.mode == "paged"),
        cuda_graph=args.cuda_graph and args.mode == "paged",
        cuda_graph_batch_sizes=cuda_graph_batch_sizes,
        cuda_graph_max_blocks=args.cuda_graph_max_blocks,
        disable_radix_cache=args.disable_radix_cache,
        cpu_cache_size_gb=args.cpu_cache_size_gb if args.mode == "paged" else 0.0,
        hicache_overlap=args.hicache_overlap and args.mode == "paged",
    )
    sched = Scheduler(
        engine=engine,
        max_running=args.max_running,
        mode=args.mode,
        prefill_chunk_size=args.prefill_chunk_size if args.mode == "paged" else 0,
        enable_retraction=args.enable_retraction and args.mode == "paged",
    )

    # Wire up the server module globals
    srv.engine = engine
    srv.scheduler = sched
    srv.model_id = args.model

    # Start scheduler background thread
    sched.start()

    logger.info("Starting server on %s:%d", args.host, args.port)
    try:
        uvicorn.run(srv.app, host=args.host, port=args.port, log_level="info")
    finally:
        sched.stop()


if __name__ == "__main__":
    main()
