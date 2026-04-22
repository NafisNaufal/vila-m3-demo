#!/usr/bin/env bash
# Setup script for VILA-M3 demo on CUDA 11.8 (driver >= 450.80.02, e.g. 470.x)
# Usage: bash setup_cu118.sh [conda_env_name]
#
# Differences from the stock environment_setup.sh:
#   - Installs torch 2.3.0+cu118 / torchvision 0.18.0+cu118 from the PyTorch wheel index
#   - Uses the flash-attn cu118 pre-built wheel (not cu122)
#   - Installs VILA with --no-deps to prevent PyPI from clobbering the cu118 torch
#   - Installs deepspeed without compiling CUDA ops (not needed for inference)
#   - Upgrades bitsandbytes to 0.43+ (0.41.0 has cu118 detection bugs)
#   - Installs Gradio 4.x (demo code uses 4.x API, pyproject.toml pin is overridden)

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VILA_DIR="$REPO_ROOT/thirdparty/VILA"

# ── 0. Optional conda env ────────────────────────────────────────────────────
CONDA_ENV=${1:-""}
if [ -n "$CONDA_ENV" ]; then
    eval "$(conda shell.bash hook)"
    conda create -n "$CONDA_ENV" python=3.10 -y
    conda activate "$CONDA_ENV"
fi

pip install --upgrade pip setuptools wheel

# ── 1. PyTorch cu118 ────────────────────────────────────────────────────────
# Must happen BEFORE `pip install -e .` so the cu118 build is not overwritten.
echo ">>> Installing PyTorch 2.3.0+cu118 ..."
pip install \
    torch==2.3.0+cu118 \
    torchvision==0.18.0+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ── 2. Flash Attention cu118 ────────────────────────────────────────────────
echo ">>> Installing Flash Attention 2.5.8 (cu118) ..."
pip install \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

# ── 3. VILA package — install WITHOUT deps so pip can't overwrite torch ──────
echo ">>> Installing VILA (no-deps) ..."
cd "$VILA_DIR"
pip install -e . --no-deps

# Now install VILA's deps manually, skipping torch/torchvision (already installed).
# bitsandbytes is bumped to 0.43.1 — 0.41.0 has cu118 detection issues.
# deepspeed is installed without CUDA op compilation (not needed for inference).
echo ">>> Installing VILA dependencies ..."
pip install \
    "transformers==4.37.2" \
    "tokenizers>=0.15.2" \
    "sentencepiece==0.1.99" \
    "shortuuid" \
    "accelerate==0.27.2" \
    "peft>=0.9.0" \
    "bitsandbytes>=0.43.1" \
    "pydantic<2,>=1" \
    "markdown2[all]" \
    "numpy==1.26.0" \
    "scikit-learn==1.2.2" \
    "requests" \
    "httpx==0.24.0" \
    "uvicorn" \
    "fastapi" \
    "einops==0.6.1" \
    "einops-exts==0.0.4" \
    "timm==0.9.12" \
    "openpyxl==3.1.2" \
    "decord==0.6.0" \
    "datasets==2.16.1" \
    "openai==1.8.0" \
    "webdataset==0.2.86" \
    "nltk==3.3" \
    "opencv-python==4.8.0.74" \
    "tyro" \
    "s2wrapper@git+https://github.com/bfshi/scaling_on_scales"

# deepspeed: DS_BUILD_OPS=0 skips CUDA kernel compilation (inference-only is fine)
echo ">>> Installing DeepSpeed (no CUDA ops) ..."
DS_BUILD_OPS=0 pip install deepspeed==0.9.5

# pytorchvideo needs fvcore which can be tricky; install separately
pip install pytorchvideo==0.1.5 || echo "WARNING: pytorchvideo install failed — video inputs won't work but image demo is unaffected."

# ── 4. Patch HuggingFace Transformers ───────────────────────────────────────
echo ">>> Patching transformers with VILA overrides ..."
pip install "git+https://github.com/huggingface/transformers@v4.37.2"
SITE=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv "$VILA_DIR/llava/train/transformers_replace/"* "$SITE/transformers/"
cp -rv "$VILA_DIR/llava/train/deepspeed_replace/"* "$SITE/deepspeed/"

# ── 5. Demo dependencies ─────────────────────────────────────────────────────
echo ">>> Installing demo dependencies ..."
cd "$REPO_ROOT"
# gradio 4.x — intentionally overrides the 3.35.2 pin in VILA's pyproject.toml
pip install \
    python-dotenv \
    "gradio>=4.20.0,<5.0.0" \
    "monai[nibabel,pynrrd,skimage,fire,ignite]" \
    torchxrayvision \
    huggingface_hub \
    colored

# ── 6. Download expert model checkpoints ────────────────────────────────────
echo ">>> Downloading TorchXRayVision weights ..."
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
    wget -nc "$BASE_URL/$fname" -O "$HOME/.torchxrayvision/models_data/$fname"
done

echo ">>> Downloading MONAI bundles ..."
mkdir -p "$HOME/.cache/torch/hub/bundle"
python -m monai.bundle download vista3d --version 0.5.4 --bundle_dir "$HOME/.cache/torch/hub/bundle"
python -m monai.bundle download brats_mri_segmentation --version 0.5.2 --bundle_dir "$HOME/.cache/torch/hub/bundle"
unzip -n "$HOME/.cache/torch/hub/bundle/vista3d_v0.5.4.zip" -d "$HOME/.cache/torch/hub/bundle/vista3d_v0.5.4"

echo ""
echo "✓ Setup complete. To run the demo:"
echo "  cd $REPO_ROOT/m3/demo && python gradio_m3.py"
