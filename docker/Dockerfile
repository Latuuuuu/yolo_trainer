# Base image ships CUDA + PyTorch + Ultralytics, so we only add the extras.
# `latest` can introduce breaking changes: once your project works, pin a
# version here (e.g. 8.4.33) and rebuild.
ARG ULTRALYTICS_TAG=latest
FROM ultralytics/ultralytics:${ULTRALYTICS_TAG}

# Command-line tools + the X11/OpenGL libraries that cv2.imshow needs
RUN --mount=type=cache,target=/var/cache/apt,sharing=private \
    apt-get update && apt-get install -y --no-install-recommends \
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
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Export/deployment extras. opencv-python replaces the headless build from the
# base image, otherwise cv2.imshow (predict.py --show) does nothing.
RUN pip install --no-cache-dir -U ultralytics \
 && pip install --no-cache-dir \
    "onnx>=1.12.0" "onnxslim>=0.1.71" onnxruntime-gpu \
    albumentations thop \
    "tensorrt-cu12>=7.0.0,!=10.1.0" \
 && pip uninstall -y opencv-python-headless \
 && pip install --no-cache-dir opencv-python

# The container runs as your host user (see compose.yaml), who does not own /root,
# so give Ultralytics and matplotlib writable config directories.
ENV YOLO_CONFIG_DIR=/.config/Ultralytics \
    MPLCONFIGDIR=/tmp/matplotlib
RUN mkdir -p /.config/Ultralytics /tmp/matplotlib

# Point Ultralytics at the mounted repository: auto-downloaded weights (e.g.
# yolo11n.pt) land in weights/, not inside the image, which the host user
# cannot write to.
RUN yolo settings weights_dir=/usr/src/weights datasets_dir=/usr/src/datasets runs_dir=/usr/src/runs \
 && chmod -R 777 /.config /tmp/matplotlib

# The repository itself is bind-mounted here by compose.yaml — nothing is
# copied into the image, so your edits take effect without rebuilding.
WORKDIR /usr/src

CMD ["/bin/bash"]
