# Use a high-performance PyTorch base image with CUDA and developer tools
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Download via the classic LFS/HTTP path rather than HuggingFace's Xet backend.
# hf_xet dies parsing content hashes ("Unable to parse string as hex hash value")
# against Xet-backed repos such as google/gemma-3; the LFS path fetches the same
# files without incident. HF_HUB_ENABLE_HF_TRANSFER is gone because hf_transfer
# is no longer used at all -- setting it only produced a deprecation warning.
ENV HF_HUB_DISABLE_XET=1
ENV MODELS_ROOT=/workspace/models

# Install system dependencies (ffmpeg is crucial for video/audio decoding/encoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    pkg-config \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Pin the torch stack before installing ltx-*. The base image ships torch and
# torchaudio 2.4.0; ltx-core requires torch~=2.7, so pip upgrades torch but
# leaves torchaudio untouched (its bare requirement is already satisfied). The
# stale torchaudio is then built against a libtorch it no longer has, and
# importing it dies on an undefined torch::autograd symbol. These three
# versions are the matched set that upstream's uv.lock resolves to.
RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchaudio==2.9.1 \
    torchvision==0.24.1

# Install sub-packages from the official Lightricks LTX-2 monorepo, pinned to
# the commit whose uv.lock produced the torch versions above.
ARG LTX_REF=9377758131b1ffde4b7f766804590a6617bf2ab9
RUN pip install --no-cache-dir "git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-core"
RUN pip install --no-cache-dir "git+https://github.com/Lightricks/LTX-2.git@${LTX_REF}#subdirectory=packages/ltx-pipelines"

# Install other serverless dependencies
RUN pip install \
    runpod \
    boto3 \
    huggingface_hub \
    av \
    tqdm \
    pillow \
    openimageio \
    "cloudpickle>=3.1"

# Create app directory
WORKDIR /app

# Copy handler and download scripts
COPY handler.py download_models.py staging.py ./

# Expose target models path, plus the local-disk cache the handler stages
# weights into. /local-cache must live on the container disk, never on the
# network volume -- staging exists precisely to get the reads off that volume.
RUN mkdir -p /workspace/models /local-cache

# Set the default entrypoint to runpod serverless handler
CMD ["python", "-u", "handler.py"]