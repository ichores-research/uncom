FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04

# Set noninteractive mode for clean installs
ENV DEBIAN_FRONTEND=noninteractive

# Install build essentials and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
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
    && rm -rf /var/lib/apt/lists/*



# Install Python 3.9.21 from source
RUN wget https://www.python.org/ftp/python/3.9.21/Python-3.9.21.tgz && \
    tar xzf Python-3.9.21.tgz && \
    cd Python-3.9.21 && \
    ./configure --enable-optimizations && \
    make -j$(nproc) && \
    make altinstall && \
    cd .. && rm -rf Python-3.9.21 Python-3.9.21.tgz

# Create symlinks
RUN ln -sf /usr/local/bin/python3.9 /usr/bin/python && \
    ln -sf /usr/local/bin/python3.9 /usr/bin/python3 && \
    ln -sf /usr/local/bin/pip3.9 /usr/bin/pip

# Set working directory
WORKDIR /root

# Upgrade pip, setuptools, wheel to specified versions
RUN pip install --upgrade pip==25.0 setuptools==75.8.0 wheel==0.45.1

# Install RAPIDS packages
RUN pip install --no-cache-dir --extra-index-url https://pypi.nvidia.com \
    cuml-cu11==24.8.0 \
    cudf-cu11==24.8.3 \
    cuda-python==11.8.6 \
    cubinlinker-cu11==0.3.0.post2

# Copy requirements file
COPY requirements.txt .

# Install PyTorch
RUN pip install torch==2.0.0+cu117 torchvision==0.15.1 torchmetrics==0.10.3 \
    --extra-index-url https://download.pytorch.org/whl/cu117

# Install dinov2
RUN git clone https://github.com/facebookresearch/dinov2.git && \
    cd dinov2 && \
    pip install -e . --no-deps

# Install remaining requirements
RUN pip install --no-cache-dir -r requirements.txt

# Override the NVIDIA entrypoint to prevent path issues
ENTRYPOINT []

# Default command
CMD ["python"]