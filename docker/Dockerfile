FROM ultralytics/ultralytics:latest

# Install common tools
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y \
    curl \
    git \
    sudo \
    usbutils \
    v4l-utils \
    htop \
    iputils-ping \
    wget \
    net-tools \
    tmux \
    vim \
    wget \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir -U ultralytics
RUN pip install --no-cache-dir onnx>=1.12.0 onnxslim>=0.1.71 onnxruntime-gpu
RUN pip install --no-cache-dir albumentations thop 
RUN pip uninstall -y opencv-python-headless
RUN pip install opencv-python 

RUN pip install --no-cache-dir "tensorrt-cu12>=7.0.0,!=10.1.0"

RUN mkdir -p /.config/Ultralytics && chmod -R 777 /.config

WORKDIR /usr/src/app

COPY . .

ENV YOLO_CONFIG_DIR=/.config/Ultralytics

CMD ["/bin/bash"]