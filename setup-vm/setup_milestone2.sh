#!/usr/bin/env bash
# Setup script for milestone 2 on a CUDA GPU instance (AWS g5.2xlarge or GCP L4).
# Idempotent — safe to re-run.
#
# Assumes:
#  - You're inside the cloned repo (cwd == repo root).
#  - The OS image already has CUDA + PyTorch (Deep Learning AMI / Deep Learning VM).
#  - Python 3.10+ is the system python.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "── 1. Verify GPU ────────────────────────────────────────────────"
if ! command -v nvidia-smi >/dev/null; then
    echo "ERROR: nvidia-smi not found. Are you on a GPU instance?" >&2
    exit 1
fi
nvidia-smi -L
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

echo
echo "── 2. Pick a Python ─────────────────────────────────────────────"
# Prefer the deep-learning AMI's pre-built python (already linked against the
# right CUDA wheels). Fall back to system python3.
if [[ -x /opt/pytorch/bin/python ]]; then
    PYTHON=/opt/pytorch/bin/python
elif command -v python3.11 >/dev/null; then
    PYTHON=$(command -v python3.11)
elif command -v python3.10 >/dev/null; then
    PYTHON=$(command -v python3.10)
else
    PYTHON=$(command -v python3)
fi
echo "Using $PYTHON ($($PYTHON --version))"

echo
echo "── 3. Create venv ───────────────────────────────────────────────"
# Ubuntu base images often strip ensurepip — make sure python3-venv is
# present so `python -m venv` works.
if ! $PYTHON -c "import ensurepip" 2>/dev/null; then
    PY_MIN=$($PYTHON -c "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')")
    echo "Installing python${PY_MIN}-venv (ensurepip missing) …"
    sudo apt-get update -qq
    sudo apt-get install -y -qq "python${PY_MIN}-venv"
fi
# A failed earlier attempt may have left a half-built .venv (directory
# exists but bin/activate doesn't). Test for the activate script, not
# just the directory.
if [[ ! -f .venv/bin/activate ]]; then
    rm -rf .venv
    $PYTHON -m venv .venv --system-site-packages
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -U pip wheel setuptools

echo
echo "── 4. Install CUDA-enabled PyTorch ──────────────────────────────"
# GCP "common-cu*" images ship CUDA + driver but NOT PyTorch. We pin to
# the cu124 wheel — driver 580/CUDA 12.9 is forward-compatible with the
# cu124 toolkit, and cu124 is the most-tested pytorch wheel for current
# flash-attn releases.
if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "Installing torch (cu124 wheel) — this can take a few minutes …"
    pip install --index-url https://download.pytorch.org/whl/cu124 torch
fi
python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: torch still can't see the GPU. Inspect `nvidia-smi` and the cu124 install above.", file=sys.stderr)
    sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
print("total VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))
PY

echo
echo "── 5. Install miniengine + bench deps ───────────────────────────"
# Skip torch in the editable install since we already installed the CUDA wheel.
pip install --no-deps -e .
pip install transformers safetensors huggingface_hub fastapi "uvicorn[standard]" pydantic aiohttp numpy tqdm datasets requests

echo
echo "── 6. Install flash-attn (this is the slow part: ~10–25 min) ────"
# --no-build-isolation: build against the venv's torch instead of pulling a
# fresh torch into the build env (which would mismatch CUDA versions).
# Verify the existing install actually IMPORTS — a corrupt-ABI wheel
# can be "installed" per pip but unimportable.
if python -c "import flash_attn" 2>/dev/null; then
    echo "flash-attn already installed and importable: $(python -c 'import flash_attn; print(flash_attn.__version__)')"
else
    # flash-attn's setup.py imports ninja + packaging + psutil at metadata
    # time; --no-build-isolation means we have to provide them ourselves.
    pip install ninja packaging psutil
    # If a broken wheel is sitting on disk, pip's "already satisfied"
    # check would skip the reinstall even with --no-binary. Uninstall
    # first so the source build actually runs.
    pip uninstall -y flash-attn 2>/dev/null || true
    # --no-binary forces a source build so the resulting .so links
    # against the torch we installed above. Prebuilt PyPI wheels are
    # built against specific (torch_version, cuda, cxx11_abi) tuples
    # and frequently mismatch the local torch's libstdc++ ABI, giving
    # a runtime "undefined symbol c10::Error..." import error.
    # NOTE: quote the version spec — unquoted, bash parses `>=2.5.0`
    # as a stdout redirect to the literal filename `=2.5.0`.
    pip install 'flash-attn>=2.5.0' --no-build-isolation --no-binary flash-attn
fi
python -c "import flash_attn; print('flash-attn:', flash_attn.__version__)"

echo
echo "── 7. Smoke-test pool unit tests (CPU, fast) ────────────────────"
pip install pytest >/dev/null
PYTHONPATH=. pytest tests/test_kv_memory_pool.py -q

echo
echo "── 8. Smoke-test model load (downloads Qwen3-8B if first time) ──"
python - <<'PY'
import torch
from miniengine.engine import Engine
from miniengine.core import Request, SamplingParams

print("Loading Qwen3-8B in paged mode …")
eng = Engine(
    model_path="Qwen/Qwen3-8B",
    dtype=torch.bfloat16,
    device="cuda",
    mode="paged",
    page_size=32,
    mem_fraction_static=0.85,
    torch_compile=False,
)
print("KV pool pages:", eng.kv_pool.num_pages, "free:", eng.kv_pool.num_free)

# Tiny generation: one short prompt → one token via paged_prefill_batch.
ids = eng.tokenize_messages([{"role": "user", "content": "Say hi in 3 words."}])
req = Request(request_id="smoke", input_ids=ids,
              sampling_params=SamplingParams(max_new_tokens=8, temperature=0.0))
toks = eng.paged_prefill_batch([req])
print("first token:", toks, "(", eng.decode_token(toks[0]), ")")
for _ in range(3):
    toks = eng.paged_decode_step([req])
    print(" decode:", toks, "(", eng.decode_token(toks[0]), ")")
print("smoke OK")
PY

echo
echo "── DONE ─────────────────────────────────────────────────────────"
echo "Next:  bash setup-vm/run_benchmarks.sh"
