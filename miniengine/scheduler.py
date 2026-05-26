"""
Request scheduler — the core orchestrator of the serving engine.

The scheduler sits between the HTTP server and the model engine:

    Server  ──add_request()──▶  Scheduler  ──prefill/decode──▶  Engine
      ▲                            │
      └─── token_queue (stream) ◄──┘

It runs in a background thread, repeatedly calling step() which:
  1. Admits waiting requests and prefills them  (WAITING → RUNNING)
  2. Runs one decode step on every running request
  3. Retires finished requests                    (RUNNING → FINISHED)

"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque

from miniengine.core import Request, RequestStatus, TokenOutput
from miniengine.engine import Engine
from miniengine.kv_memory_pool import KVOutOfMemory

logger = logging.getLogger(__name__)


class Scheduler:
    """
    FCFS scheduler with two modes:

      baseline : process one request to completion before the next.
      batched  : iteration-level batching — admit + prefill many requests,
                 then advance all running requests by one token in a
                 single batched forward pass.  New requests can join the
                 batch the same step they finish prefill.

    Public API (thread-safe):
        add_request(req)   — enqueue a new request
        start()            — launch the background scheduling loop
        stop()             — gracefully shut down
    """

    def __init__(
        self,
        engine: Engine,
        max_running: int = 16,
        mode: str = "batched",
        prefill_chunk_size: int = 0,
        enable_retraction: bool = False,
    ):
        self.engine = engine
        self.max_running = max_running
        self.mode = mode
        self.prefill_chunk_size = prefill_chunk_size
        self.enable_retraction = enable_retraction

        # Queues
        self.waiting: deque[Request] = deque()
        self.running: list[Request] = []

        # Chunked-prefill state (milestone 3 Part A). When non-None, a
        # request is being chunked across multiple scheduler steps; new
        # admissions are paused until it finishes.
        self._prefilling: Request | None = None

        # Thread control
        self._lock = threading.Lock()
        self._running_flag = False
        self._thread: threading.Thread | None = None

        # Stats
        self.total_finished: int = 0
        self.total_generated_tokens: int = 0
        self.total_retractions: int = 0

    # ── Public API (thread-safe) ────────────────────────────────────────

    def add_request(self, request: Request) -> None:
        """Enqueue a request for scheduling."""
        with self._lock:
            self.waiting.append(request)
            logger.info(
                "Enqueued request %s  (prompt_len=%d, waiting=%d)",
                request.request_id,
                request.num_input_tokens,
                len(self.waiting),
            )

    def start(self) -> None:
        """Start the scheduler loop in a background daemon thread."""
        self._running_flag = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Scheduler started")

    def stop(self) -> None:
        """Signal the scheduler to stop and wait for the thread to join."""
        self._running_flag = False
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        logger.info("Scheduler stopped")

    # ── Main loop ───────────────────────────────────────────────────────

    def _loop(self) -> None:
        while self._running_flag:
            has_work = bool(self.waiting) or bool(self.running)
            if not has_work:
                time.sleep(0.005)  # idle sleep to avoid busy-waiting
                continue
            try:
                self.step()
            except Exception:
                logger.exception("Scheduler step failed")

    # ── Scheduling step ─────────────────────────────────────────────────

    def step(self) -> list[Request]:
        """
        One scheduling iteration.  Behaviour depends on self.mode.

        Returns list of requests that finished in this step.
        """
        if self.mode == "baseline":
            return self._step_baseline()
        if self.mode == "paged":
            return self._step_paged()
        return self._step_batched()

    def _step_baseline(self) -> list[Request]:
        """One request to completion per step. Maximally naive."""
        finished: list[Request] = []

        with self._lock:
            while self.waiting and len(self.running) < self.max_running:
                req = self.waiting.popleft()
                req.status = RequestStatus.RUNNING
                self.running.append(req)

        req.status = RequestStatus.RUNNING
        token_id = self.engine.prefill(req)
        req.output_ids.append(token_id)
        self._stream_token(req, token_id)

        while not self._check_finished(req, token_id):
            token_id = self.engine.decode_step(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)

        # ── Phase 3: Retire finished requests ───────────────────────────
        still_running = []
        for req in self.running:
            if self._check_finished(req, req.output_ids[-1]):
                self._finish_request(req, finished)
            else:
                still_running.append(req)
        self.running = still_running

        return finished

    def _step_batched(self) -> list[Request]:
        """
        Iteration-level batched step:
          Phase 1 — admit waiting requests and prefill them (per-request).
          Phase 2 — batched decode: one token for every running request.
        Newly prefilled requests join the decode batch in the same step.
        """
        finished: list[Request] = []

        # ── Phase 1: admit + prefill ────────────────────────────────────
        with self._lock:
            to_prefill: list[Request] = []
            while (
                self.waiting and len(self.running) + len(to_prefill) < self.max_running
            ):
                to_prefill.append(self.waiting.popleft())

        for req in to_prefill:
            req.status = RequestStatus.RUNNING
            token_id = self.engine.prefill(req)
            req.output_ids.append(token_id)
            self._stream_token(req, token_id)
            if self._check_finished(req, token_id):
                self._finish_request(req, finished)
            else:
                self.running.append(req)

        # ── Phase 2: batched decode ─────────────────────────────────────
        if self.running:
            token_ids = self.engine.batched_decode(self.running)
            still_running: list[Request] = []
            for req, token_id in zip(self.running, token_ids):
                req.output_ids.append(token_id)
                self._stream_token(req, token_id)
                if self._check_finished(req, token_id):
                    self._finish_request(req, finished)
                else:
                    still_running.append(req)
            self.running = still_running

        return finished

    def _step_paged(self) -> list[Request]:
        """
        Paged step — milestone-3 cache + chunked-prefill + retraction.

        Phase 1: admit / advance chunked prefill.
            * If a chunked prefill is in flight (``self._prefilling``),
              advance one chunk this step. No new admissions.
            * Else, when ``prefill_chunk_size > 0``: start chunking the
              head of the waiting queue (lookup, allocate prompt pages,
              do the first chunk).
            * Else (``prefill_chunk_size == 0``): admit a batch using the
              milestone-2 packed prefill path (but cache-aware and lazy
              alloc — see ``engine.paged_prefill_batch``).

        Phase 2: paged decode for every running request. Lazy-alloc may
            raise ``KVOutOfMemory``; if retraction is enabled, evict a
            victim and retry.
        """
        finished: list[Request] = []
        engine = self.engine
        pool = engine.kv_pool
        assert pool is not None, "paged mode requires engine.kv_pool"

        # ── Phase 1: admit / advance prefill ───────────────────────────
        if self.prefill_chunk_size > 0:
            self._step_paged_admit_chunked(finished)
        else:
            self._step_paged_admit_packed(finished)

        # ── Phase 2: decode (with optional retraction retry loop) ──────
        if self.running:
            self._step_paged_decode(finished)

        return finished

    # ── Admission helpers ───────────────────────────────────────────────

    def _step_paged_admit_packed(self, finished: list[Request]) -> None:
        """Milestone-2-style admission: packed multi-request prefill.

        With milestone-3 lazy alloc, admission gating is by *prompt* pages
        only (no max_new_tokens reservation).
        """
        engine = self.engine
        pool = engine.kv_pool

        with self._lock:
            to_prefill: list[Request] = []
            free_after_admit = pool.num_free + pool.num_evictable
            while (
                self.waiting
                and len(self.running) + len(to_prefill) < self.max_running
            ):
                req = self.waiting[0]
                # Best-effort: estimate prompt pages BEFORE cache lookup.
                # The actual lookup happens inside paged_prefill_batch
                # and may reduce the page demand; this gate just avoids
                # admitting requests we definitely can't service.
                need = pool.pages_needed(req.num_input_tokens)
                if need > free_after_admit:
                    break
                self.waiting.popleft()
                to_prefill.append(req)
                free_after_admit -= need

        if not to_prefill:
            return

        for req in to_prefill:
            req.status = RequestStatus.RUNNING
        try:
            token_ids = engine.paged_prefill_batch(to_prefill)
        except KVOutOfMemory:
            # Genuine prefill OOM: push the requests back to waiting so a
            # later step (with more freed pool capacity) can try again.
            logger.warning(
                "KV OOM during packed prefill of %d requests; deferring.",
                len(to_prefill),
            )
            with self._lock:
                for req in reversed(to_prefill):
                    req.status = RequestStatus.WAITING
                    self.waiting.appendleft(req)
            return

        for req, tok in zip(to_prefill, token_ids):
            req.output_ids.append(tok)
            self._stream_token(req, tok)
            if self._check_finished(req, tok):
                self._finish_request(req, finished)
            else:
                self.running.append(req)

    def _step_paged_admit_chunked(self, finished: list[Request]) -> None:
        """Single-request chunked prefill (milestone 3 Part A).

        Each step processes one chunk of ``prefill_chunk_size`` Q-tokens.
        Other already-running requests continue decoding in parallel
        (handled in Phase 2 of ``_step_paged``).
        """
        engine = self.engine

        if self._prefilling is None:
            # Try to start a new chunked prefill.
            with self._lock:
                if not self.waiting:
                    return
                req = self.waiting.popleft()
            req.status = RequestStatus.RUNNING
            try:
                engine.start_paged_prefill(req)
            except KVOutOfMemory:
                logger.warning(
                    "KV OOM starting chunked prefill of %s; re-queuing.",
                    req.request_id,
                )
                with self._lock:
                    req.status = RequestStatus.WAITING
                    self.waiting.appendleft(req)
                return
            self._prefilling = req

        # Advance one chunk of the current prefill.
        try:
            tok = engine.paged_prefill_chunk(self._prefilling, self.prefill_chunk_size)
        except KVOutOfMemory:
            # Pool empty mid-prefill. Best effort: defer this request back
            # to waiting. Its already-written pages stay allocated to it
            # so the next attempt can pick up where it left off — but for
            # simplicity (and to avoid stale-state edge cases) we retract
            # it entirely and let it re-prefill on next admission.
            logger.warning(
                "KV OOM in chunked prefill of %s; retracting and re-queuing.",
                self._prefilling.request_id,
            )
            victim = self._prefilling
            engine.retract_paged_request(victim)
            victim.output_ids = []
            victim.status = RequestStatus.WAITING
            with self._lock:
                self.waiting.appendleft(victim)
            self._prefilling = None
            return

        if tok is None:
            return  # mid-chunk; keep going next step

        # Final chunk landed — this request has its first generated token.
        req = self._prefilling
        self._prefilling = None
        req.output_ids.append(tok)
        self._stream_token(req, tok)
        if self._check_finished(req, tok):
            self._finish_request(req, finished)
        else:
            self.running.append(req)

    def _step_paged_decode(self, finished: list[Request]) -> None:
        """Run one decode step on every running request, with retraction.

        If the engine raises ``KVOutOfMemory`` mid-decode and retraction
        is enabled, we evict a victim and retry. If no victim is
        available (everyone is at cache_len==0, or only the chunked-
        prefill request is left), we drop the step and log.
        """
        engine = self.engine
        while True:
            try:
                token_ids = engine.paged_decode_step(self.running)
                break
            except KVOutOfMemory:
                if not self.enable_retraction:
                    logger.error(
                        "KV OOM during decode and --enable-retraction is off; "
                        "decode step aborted (%d running).", len(self.running)
                    )
                    return
                if not self._retract_one_victim():
                    logger.error(
                        "KV OOM and no eligible retraction victim; "
                        "decode step aborted (%d running).", len(self.running)
                    )
                    return
                # Loop and retry with one fewer running request.

        still_running: list[Request] = []
        for req, tok in zip(self.running, token_ids):
            req.output_ids.append(tok)
            self._stream_token(req, tok)
            if self._check_finished(req, tok):
                self._finish_request(req, finished)
            else:
                still_running.append(req)
        self.running = still_running

    def _retract_one_victim(self) -> bool:
        """Evict a running request back to the waiting queue to free pages.

        Policy: youngest-first (latest ``arrival_time``); tie-break by
        largest remaining work (``max_new_tokens - num_output_tokens``).
        The in-flight chunked-prefill request is never eligible — it lives
        in ``self._prefilling``, not ``self.running``.
        """
        if not self.running:
            return False
        # Prefer victims that actually have pages to free.
        candidates = [r for r in self.running if r.page_table]
        if not candidates:
            return False
        victim = max(
            candidates,
            key=lambda r: (
                r.arrival_time,
                r.sampling_params.max_new_tokens - r.num_output_tokens,
            ),
        )
        self.engine.retract_paged_request(victim)
        victim.output_ids = []
        victim.status = RequestStatus.WAITING
        self.running.remove(victim)
        with self._lock:
            self.waiting.appendleft(victim)
        self.total_retractions += 1
        logger.info(
            "Retracted %s back to waiting queue (total retractions=%d).",
            victim.request_id,
            self.total_retractions,
        )
        return True

    # ── Helpers ─────────────────────────────────────────────────────────

    def _check_finished(self, req: Request, token_id: int) -> bool:
        """Decide whether a request should stop generating."""
        if req.is_finished:
            return True
        if self.engine.is_stop_token(token_id):
            return True
        return False

    def _stream_token(self, req: Request, token_id: int) -> None:
        """Push a generated token into the request's streaming queue."""
        text = self.engine.decode_token(token_id)
        req.token_queue.put(
            TokenOutput(token_id=token_id, token_text=text, finished=False)
        )

    def _finish_request(self, req: Request, finished_list: list[Request]) -> None:
        """Mark a request as finished and free its resources."""
        req.status = RequestStatus.FINISHED
        if self.mode == "paged":
            # Cache-aware free: inserts the full (prompt+output) into the
            # radix cache for future multi-turn hits, decrements lock-ref,
            # returns redundant + tail pages to the pool.
            self.engine.free_paged_request(req)
        req.kv_cache = None  # release GPU memory (baseline/batched paths)
        req.token_queue.put(TokenOutput(token_id=-1, token_text="", finished=True))
        finished_list.append(req)

        self.total_finished += 1
        self.total_generated_tokens += req.num_output_tokens
        logger.info(
            "Finished request #%d %s  (output_len=%d, hit=%d, running=%d, waiting=%d)",
            self.total_finished,
            req.request_id,
            req.num_output_tokens,
            req.cache_hit_tokens,
            len(self.running),
            len(self.waiting),
        )
