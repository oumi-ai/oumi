#!/bin/bash
# Install flash-attn, preferring a prebuilt wheel over the (~25-60 min) source build.
#
# Why this exists: flash-attn publishes no binary wheel on PyPI, and the official
# GitHub release wheels lag behind new torch versions (e.g. no torch>=2.9 wheels as of
# flash-attn 2.8.3). So `pip install flash-attn` almost always compiles from source.
# The community mjun0812/flash-attention-prebuild-wheels project builds wheels across a
# wide CUDA x torch x Python matrix; this script picks the one matching the *currently
# installed* torch and falls back to a source build if no match works.
#
# Correctness model: we do not assume which versions the resolver produced. We read the
# live interpreter (Python tag, torch version, torch's CUDA build) to construct the wheel
# URL, then REQUIRE `import flash_attn` to succeed before trusting the wheel. An ABI or
# version mismatch surfaces as an ImportError ("undefined symbol") at that check, which
# triggers the source-build fallback -- so a wrong wheel can never silently pass.

set -uo pipefail

FA_VERSION="${FA_VERSION:-2.8.3}"
# Release tag of mjun0812/flash-attention-prebuild-wheels that hosts FA_VERSION wheels.
FA_WHEEL_RELEASE="${FA_WHEEL_RELEASE:-v0.9.0}"
FA_WHEEL_REPO="mjun0812/flash-attention-prebuild-wheels"

source_build() {
  echo "[flash-attn] Building from source (this can take 25-60 min)..."
  pip install -U "flash-attn==${FA_VERSION}" --no-build-isolation
}

# Derive the env coordinates from the live interpreter. If torch is missing, there is
# nothing to match a wheel against -- go straight to the source build (pip will pull
# whatever torch flash-attn needs).
if ! python -c "import torch" 2>/dev/null; then
  echo "[flash-attn] torch not importable; cannot match a prebuilt wheel."
  source_build
  exit $?
fi

# py_tag e.g. cp310 ; torch_mm e.g. 2.9 ; cuda_tag e.g. cu128 (empty if CPU/rocm torch)
read -r PY_TAG TORCH_MM CUDA_TAG <<EOF
$(python - <<'PY'
import sys, torch
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
mm = ".".join(torch.__version__.split("+")[0].split(".")[:2])  # 2.9.1 -> 2.9
cuda = torch.version.cuda or ""                                 # "12.8" or None
cuda_tag = ("cu" + cuda.replace(".", "")) if cuda else ""
print(py, mm, cuda_tag)
PY
)
EOF

echo "[flash-attn] Detected: python=${PY_TAG} torch=${TORCH_MM} cuda=${CUDA_TAG:-none}"

if [[ -z "${CUDA_TAG}" ]]; then
  echo "[flash-attn] torch has no CUDA build tag; prebuilt wheels are CUDA-only."
  source_build
  exit $?
fi

# Construct the wheel URL. The '+' in the local version label is URL-encoded as %2B.
WHEEL="flash_attn-${FA_VERSION}+${CUDA_TAG}torch${TORCH_MM}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
URL="https://github.com/${FA_WHEEL_REPO}/releases/download/${FA_WHEEL_RELEASE}/${WHEEL/+/%2B}"

echo "[flash-attn] Trying prebuilt wheel: ${URL}"
# --no-deps so the wheel cannot drag torch/cuda libs and disturb the resolved env.
if pip install --no-deps "${URL}" && python -c "import flash_attn; print('[flash-attn] import OK:', flash_attn.__version__)"; then
  echo "[flash-attn] Installed prebuilt wheel for ${CUDA_TAG}/torch${TORCH_MM}/${PY_TAG}."
  exit 0
fi

echo "[flash-attn] Prebuilt wheel missing or failed to import; falling back to source."
# Clean up a possibly half-installed wheel so the source build starts from a clean slate.
pip uninstall -y flash-attn flash_attn >/dev/null 2>&1 || true
source_build
