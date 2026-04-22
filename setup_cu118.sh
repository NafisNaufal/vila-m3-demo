#!/usr/bin/env bash
# VILA-M3 inference environment setup — CUDA 11.8
#
# Compatible with NVIDIA driver >= 450.80.02 (e.g. 470.x shows "CUDA 11.6" in nvidia-smi
# but still supports CUDA 11.8 runtime — minimum driver for cu118 is 450.80.02).
#
# Usage:
#   bash setup_cu118.sh                   # use already-activated env
#   bash setup_cu118.sh my_conda_env      # create + activate a new conda env first
#
# What this does differently from the stock environment_setup.sh:
#   - Wipes any existing torch/torchvision before installing to avoid cu130/cpu builds
#   - Installs torch==2.3.0+cu118 from the PyTorch wheel index (not PyPI)
#   - Installs VILA with --no-deps so pip can never overwrite the cu118 torch
#   - Skips flash-attn  (not required for inference; VILA falls back to standard attention)
#   - Skips deepspeed   (training only; not imported in the inference path)
#   - Installs pydantic 2.x (gradio 4.x requires it; deepspeed which needed <2 is skipped)
#   - Installs gradio 4.x  (demo code uses 4.x API; pyproject.toml pin of 3.35.2 is wrong)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VILA_DIR="$REPO_ROOT/thirdparty/VILA"

log()  { echo -e "\033[1;34m>>>\033[0m $*"; }
warn() { echo -e "\033[1;33mWARN:\033[0m $*"; }
die()  { echo -e "\033[1;31mERROR:\033[0m $*" >&2; exit 1; }

# ── 0. Optional: create & activate a conda env ───────────────────────────────
CONDA_ENV="${1:-}"
if [ -n "$CONDA_ENV" ]; then
    if ! command -v conda &>/dev/null; then
        die "conda not found. Either install conda or omit the env name and activate manually."
    fi
    eval "$(conda shell.bash hook)"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
fi

# ── 1. Sanity-check: must be Python 3.10 ─────────────────────────────────────
python3 -c "
import sys
v = sys.version_info[:2]
if v != (3, 10):
    print(f'ERROR: Python 3.10 required, got {sys.version}')
    sys.exit(1)
print(f'Python {sys.version.split()[0]} OK')
" || die "Activate a Python 3.10 environment first."

log "Upgrading pip / setuptools / wheel ..."
pip install --upgrade pip setuptools wheel

# ── 2. Remove any wrong torch build (cu130, cpu, etc.) ───────────────────────
log "Removing any existing torch / torchvision / flash-attn ..."
pip uninstall -y torch torchvision torchaudio flash-attn 2>/dev/null || true

# ── 3. Install PyTorch 2.3.0 + CUDA 11.8 ─────────────────────────────────────
log "Installing torch==2.3.0+cu118 and torchvision==0.18.0+cu118 ..."
pip install \
    torch==2.3.0+cu118 \
    torchvision==0.18.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 4. Verify CUDA works before going further ─────────────────────────────────
log "Verifying CUDA availability ..."
python3 - <<'PYEOF'
import sys, torch
print(f"  torch        : {torch.__version__}")
print(f"  torch.cuda   : {torch.version.cuda}")
if not torch.cuda.is_available():
    print("ERROR: torch.cuda.is_available() returned False.")
    print("       Driver version may not support CUDA 11.8 (need driver >= 450.80.02).")
    sys.exit(1)
print(f"  GPU          : {torch.cuda.get_device_name(0)}")
print("  CUDA check   : PASSED")
PYEOF

# ── 5. Install VILA in editable mode, no-deps ────────────────────────────────
# --no-deps is critical: prevents pip from reinstalling torch==2.3.0 from PyPI
# (the plain PyPI build has no CUDA and would silently break GPU inference).
log "Installing VILA package (editable, no-deps) ..."
cd "$VILA_DIR"
pip install -e . --no-deps

# ── 6. Install VILA runtime dependencies ─────────────────────────────────────
# Intentionally omitted vs pyproject.toml:
#   torch / torchvision  already installed (cu118)
#   flash-attn           not needed for inference
#   deepspeed            training-only; conflicts with pydantic 2.x (gradio 4.x needs 2.x)
#   gradio==3.35.2       overridden below; demo uses 4.x API
#   gradio_client==0.2.9 same
#   pre-commit / pytest  dev tools
#   pywsd==1.2.4         eval-only
#   transformers         installed from git in step 7 to ensure exact patched version
log "Installing VILA dependencies ..."
pip install \
    "tokenizers>=0.15.2" \
    "sentencepiece==0.1.99" \
    "shortuuid" \
    "accelerate==0.27.2" \
    "peft>=0.9.0" \
    "bitsandbytes>=0.43.1" \
    "pydantic>=2.0" \
    "markdown2[all]" \
    "numpy==1.26.0" \
    "scikit-learn==1.2.2" \
    "requests" \
    "httpx>=0.24.0" \
    "uvicorn" \
    "fastapi" \
    "einops==0.6.1" \
    "einops-exts==0.0.4" \
    "timm==0.9.12" \
    "openpyxl==3.1.2" \
    "datasets==2.16.1" \
    "openai>=1.8.0" \
    "webdataset==0.2.86" \
    "nltk==3.3" \
    "opencv-python==4.8.0.74" \
    "tyro" \
    "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"

# video deps — used by VILA's video pipeline but not the medical image demo
pip install "decord==0.6.0" \
    || warn "decord install failed — video inputs won't work (OK for image demo)"
pip install "pytorchvideo==0.1.5" \
    || warn "pytorchvideo install failed — video inputs won't work (OK for image demo)"

# ── 7. Install & patch HuggingFace Transformers ───────────────────────────────
# The VILA repo patches transformers internals (attention, generation).
# We install from the exact git tag then copy the overrides on top.
log "Installing patched transformers 4.37.2 ..."
pip install "git+https://github.com/huggingface/transformers@v4.37.2"
SITE=$(python3 -c 'import site; print(site.getsitepackages()[0])')
cp -rv "$VILA_DIR/llava/train/transformers_replace/"* "$SITE/transformers/"
# deepspeed_replace is intentionally skipped — deepspeed is not installed

# ── 8. Demo dependencies ─────────────────────────────────────────────────────
log "Installing demo dependencies (gradio 4.x, monai, torchxrayvision) ..."
cd "$REPO_ROOT"
pip install \
    python-dotenv \
    "gradio>=4.20.0,<5.0.0" \
    "monai[nibabel,pynrrd,skimage,fire,ignite]" \
    torchxrayvision \
    "huggingface_hub>=0.20.0" \
    colored

# ── 9. Download TorchXRayVision weights ──────────────────────────────────────
log "Downloading TorchXRayVision model weights ..."
mkdir -p "$HOME/.torchxrayvision/models_data/"
BASE_URL="https://github.com/mlmed/torchxrayvision/releases/download/v1"
for fname in \
    nih-pc-chex-mimic_ch-google-openi-kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    chex-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    mimic_ch-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    mimic_nb-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    nih-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    pc-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    kaggle-densenet121-d121-tw-lr001-rot45-tr15-sc15-seed0-best.pt \
    pc-nih-rsna-siim-vin-resnet50-test512-e400-state.pt; do
    wget -nc "$BASE_URL/$fname" \
         -O "$HOME/.torchxrayvision/models_data/$fname" \
        || warn "Failed to download $fname (will be re-attempted on next run)"
done

# ── 10. Download MONAI expert bundles ─────────────────────────────────────────
log "Downloading MONAI bundles (vista3d, brats_mri_segmentation) ..."
mkdir -p "$HOME/.cache/torch/hub/bundle"
python3 -m monai.bundle download vista3d \
    --version 0.5.4 \
    --bundle_dir "$HOME/.cache/torch/hub/bundle"
python3 -m monai.bundle download brats_mri_segmentation \
    --version 0.5.2 \
    --bundle_dir "$HOME/.cache/torch/hub/bundle"
unzip -n "$HOME/.cache/torch/hub/bundle/vista3d_v0.5.4.zip" \
    -d "$HOME/.cache/torch/hub/bundle/vista3d_v0.5.4"

# ── 11. Final environment summary ─────────────────────────────────────────────
log "Final check ..."
python3 - <<'PYEOF'
import torch
print(f"  torch        : {torch.__version__}")
print(f"  CUDA runtime : {torch.version.cuda}")
print(f"  GPU count    : {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"  GPU name     : {torch.cuda.get_device_name(0)}")
    print("  STATUS       : READY")
else:
    print("  STATUS       : WARNING — CUDA not available")
PYEOF

echo ""
log "Setup complete."
echo "  Run the demo with:"
echo "    cd $REPO_ROOT/m3/demo && python gradio_m3.py"
