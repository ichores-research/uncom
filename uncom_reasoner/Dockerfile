FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set noninteractive mode for clean installs
ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9"
ENV MMCV_WITH_OPS=1
ENV FORCE_CUDA=1
ENV CXXFLAGS="-std=c++17"

# Install deadsnakes PPA and Python 3.9
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    build-essential \
    cmake \
    ninja-build \
    wget \
    curl \
    git \
    libffi-dev \
    libssl-dev \
    libsqlite3-dev \
    libbz2-dev \
    liblzma-dev \
    libncurses5-dev \
    zlib1g-dev \
    libreadline-dev \
    tk-dev \
    xz-utils \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libjpeg-turbo8 \
    libpng16-16 \
    libjpeg-dev \
    libpng-dev \    
    ffmpeg \
    libsndfile1 \
    mosquitto-clients \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.9
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9

# Create symlinks
RUN ln -sf /usr/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip

# Set working directory
WORKDIR /root

# Upgrade pip, setuptools, wheel to specified versions
RUN pip install --upgrade pip==25.0 setuptools==75.8.0 wheel==0.45.1

# === CRITICAL: Pin NumPy FIRST to prevent NumPy 2.x ===
RUN pip install --no-cache-dir 'numpy<2,>=1.23.0'



# Install PyTorch stack with EXPLICIT CUDA 12.1 builds
# RUN pip install --no-cache-dir \
#     torch==2.1.0+cu121 \
#     torchvision==0.16.0+cu121 \
#     torchaudio==2.1.0+cu121 \
#     --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    torch==2.1.0+cu121 \
    torchvision==0.16.0+cu121 \
    torchaudio==2.1.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir torchmetrics==0.10.3 pybind11
    
# Install missing packages
RUN pip install --no-cache-dir shapely pillow paho-mqtt


# Install scipy with NumPy constraint
RUN pip install --no-cache-dir 'scipy<1.14'

# Install faiss-gpu (THIS WAS MISSING - critical for your image.py)
RUN pip install --no-cache-dir faiss-gpu

# Install RAPIDS packages
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
    cuml-cu12==24.8.* \
    cudf-cu12==24.8.* \
    cuda-python

# Install mmcv-full from prebuilt wheel
RUN pip install --no-cache-dir \
    https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl

# Install mmsegmentation
RUN pip install --no-cache-dir mmsegmentation==0.30.0

# Install other dependencies
RUN pip install --no-cache-dir \
    black==22.6.0 \
    flake8==5.0.4 \
    pylint==2.15.0 \
    xformers==0.0.18 \
    omegaconf \
    fvcore \
    iopath \
    submitit \
    mediapipe

# Install dinov2 (non-editable to avoid keeping source)
RUN git clone https://github.com/facebookresearch/dinov2.git && \
    cd dinov2 && \
    pip install . --no-deps && \
    cd .. && rm -rf dinov2

# === Install transformers and audio processing WITH NumPy constraint ===
RUN pip install --no-cache-dir \
    'transformers==4.46.3' \
    'accelerate>=0.26' \
    soundfile \
    librosa \
    sentencepiece \
    safetensors
    
# === FINAL SAFETY: Force NumPy back to 1.x in case anything upgraded it ===
RUN pip install --force-reinstall --no-cache-dir 'numpy<2,>=1.23.0'

# Add this to your Dockerfile to kill that torchvision error
RUN pip install chardet

# RUN git clone --branch v0.16.0 https://github.com/pytorch/vision.git /opt/torchvision && \
#     cd /opt/torchvision && \
#     sed -i '/^license = /d' pyproject.toml && \
#     export FORCE_CUDA=1 && \
#     export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9" && \
#     pip install --no-build-isolation . && \
#     cd /root && rm -rf /opt/torchvision

RUN pip install --no-cache-dir \
    torchvision==0.16.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121


# FORCE the linker to see the NVIDIA and Torch CUDA libs
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.9/dist-packages/torch/lib:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}"
ENV LD_PRELOAD="/usr/local/lib/python3.9/dist-packages/torch/lib/libtorch_cuda.so:/usr/local/lib/python3.9/dist-packages/torch/lib/libtorch_cpu.so:/usr/local/lib/python3.9/dist-packages/torch/lib/libc10.so:/usr/local/lib/python3.9/dist-packages/torch/lib/libc10_cuda.so"

ENTRYPOINT []
CMD ["python"]
