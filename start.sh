#!/bin/bash
# RunPod startup script for Image & Video Generator
# Required environment variables (set as RunPod secrets):
#   GITHUB_TOKEN  — personal access token for this GitHub repo
#   HF_TOKEN      — HuggingFace access token (needed for Flux.1-dev + LTX-2 model downloads)

set -e

REPO="https://${GITHUB_TOKEN}@github.com/nyankuruogega/image-video-generator.git"
APP_DIR="/workspace"

echo "=========================================="
echo " Image & Video Generator — Pod Setup"
echo "=========================================="

# ── 1. Clone or update repo ───────────────────────────────────────────────────
if [ -d "$APP_DIR/.git" ]; then
    echo "📦 Pulling latest code..."
    cd "$APP_DIR"
    git remote set-url origin "$REPO"
    git pull origin main
else
    echo "📦 Cloning repo..."
    git clone "$REPO" "$APP_DIR"
    cd "$APP_DIR"
fi

# ── 2. Install dependencies ───────────────────────────────────────────────────
if [ ! -f /workspace/.deps-installed ]; then
    echo "📦 Installing dependencies..."

    pip install -q gradio
    pip install -q Pillow imageio-ffmpeg "jinja2>=3.1.5"
    pip install -q "transformers>=4.57.0" accelerate
    pip install -q "git+https://github.com/huggingface/diffusers.git"

    # Torch (CUDA 12.4)
    pip install -q "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
        --index-url https://download.pytorch.org/whl/cu124

    # LTX-2 core packages (required for LTX-2 Distilled video generation)
    pip install -q "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core"
    pip install -q "git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines"

    # Pin torch back (ltx-core upgrades to 2.7 which breaks torchvision)
    pip install -q "torch==2.6.0+cu124" "torchvision==0.21.0+cu124" "torchaudio==2.6.0+cu124" \
        --index-url https://download.pytorch.org/whl/cu124

    # Pin transformers back (ltx-core installs 5.x which breaks Gemma3)
    pip install -q "transformers==4.57.6"

    pip install -q "imageio[ffmpeg]"

    touch /workspace/.deps-installed
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed, skipping..."
fi

# ── 3. HuggingFace authentication ─────────────────────────────────────────────
if [ -n "$HF_TOKEN" ]; then
    echo "🤗 Authenticating to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "✅ HuggingFace authenticated"
else
    echo "⚠️  HF_TOKEN not set — Flux.1-dev (gated) and LTX-2 downloads may fail"
fi

# ── 4. Start JupyterLab ───────────────────────────────────────────────────────
echo "📓 Starting JupyterLab on port 8888..."
jupyter lab \
    --ip=0.0.0.0 \
    --port=8888 \
    --no-browser \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' \
    --ServerApp.allow_origin='*' \
    --ServerApp.root_dir="$APP_DIR" \
    &>/tmp/jupyter.log &
echo "✅ JupyterLab started"

# ── 5. Start the app ──────────────────────────────────────────────────────────
echo "🚀 Starting Image & Video Generator on port 7860..."
cd "$APP_DIR"
python generate.py
