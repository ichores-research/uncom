FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu20.04

# Set noninteractive mode for clean installs
ENV DEBIAN_FRONTEND=noninteractive

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
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


# === Pin scientific stack to avoid NumPy 2 ABI problems (Python 3.9-friendly) ===
RUN pip install --no-cache-dir 'numpy<2,>=1.23' 'scipy<1.14'


RUN pip install torch==2.1.0 torchvision==0.16.0 torchmetrics==0.10.3 torchaudio==2.1.0 \
    --extra-index-url https://download.pytorch.org/whl/cu121


# Install RAPIDS packages (removed cubinlinker - it's bundled)
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
    cuml-cu12==24.8.* \
    cudf-cu12==24.8.* \
    cuda-python

# RUN pip install --no-cache-dir mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
# 1. Direct download of the exact binary wheel (bypass search/compilation)
RUN pip install --no-cache-dir \
    https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/mmcv_full-1.7.2-cp39-cp39-manylinux1_x86_64.whl

# Copy requirements file
# COPY requirements.txt .

RUN pip install mmsegmentation==0.33.0 || \
    pip install mmsegmentation==0.32.0 || \
    pip install mmsegmentation==0.31.0 || \
    pip install mmsegmentation==0.30.0

RUN pip install black==22.6.0 flake8==5.0.4 pylint==2.15.0 xformers==0.0.18 omegaconf fvcore iopath submitit mediapipe

# Install dinov2 (non-editable to avoid keeping source)
RUN git clone https://github.com/facebookresearch/dinov2.git && \
    cd dinov2 && \
    pip install . --no-deps && \
    cd .. && rm -rf dinov2

RUN pip install --no-cache-dir \
    'transformers==4.40.2' \
    'accelerate>=0.26' \
    soundfile librosa sentencepiece safetensors

# # 2. Scrub mmcv-full from the external requirements file
# RUN sed -i '/mmcv-full/d' requirements.txt
# RUN sed -i '/^torch/d;/^torchvision/d;/^torchaudio/d;/^torchmetrics/d;/cuml/d' requirements.txt

# # 3. Now the rest will install smoothly
# RUN pip install --no-cache-dir -r requirements.txt
# # Override the NVIDIA entrypoint to prevent path issues

# 5. FORCE the linker to see the NVIDIA and Torch CUDA libs
ENV LD_LIBRARY_PATH="/usr/local/lib/python3.9/dist-packages/torch/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

ENTRYPOINT []

# Default command
CMD ["python"]
