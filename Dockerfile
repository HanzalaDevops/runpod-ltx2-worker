# Use a high-performance PyTorch base image with CUDA and developer tools
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_ENABLE_HF_TRANSFER=1
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

# Install sub-packages directly from official Lightricks LTX-2 monorepo
RUN pip install git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core
RUN pip install git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines

# Install other serverless dependencies
RUN pip install \
    runpod \
    boto3 \
    huggingface_hub[hf_transfer] \
    av \
    tqdm \
    pillow \
    openimageio \
    cloudpickle>=3.1

# Create app directory
WORKDIR /app

# Copy handler and download scripts
COPY handler.py download_models.py ./

# Expose target models path
RUN mkdir -p /workspace/models

# Set the default entrypoint to runpod serverless handler
CMD ["python", "-u", "handler.py"]