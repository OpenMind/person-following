# Production Dockerfile for Person Following System (Jetson Thor + ROS 2 Jazzy)
FROM nvcr.io/nvidia/pytorch:25.10-py3

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=jazzy \
    PROJECT_ROOT=/opt/person_following \
    VIRTUAL_ENV=/opt/venv \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=""

# Prefer the UCX/UCC that ships in the base image (HPC-X), then CUDA.
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/src/tensorrt/bin:/usr/local/cuda/bin:${VIRTUAL_ENV}/bin:${PATH}

# Make sure ld.so can also find HPC-X libs
RUN if [ -d /opt/hpcx/ucx/lib ] && [ -d /opt/hpcx/ucc/lib ]; then \
      printf '%s\n' /opt/hpcx/ucx/lib /opt/hpcx/ucc/lib > /etc/ld.so.conf.d/hpcx.conf && ldconfig; \
    fi

# Enable universe (Ubuntu 24.04 uses deb822 sources file in many images)
RUN set -eux; \
    if [ -f /etc/apt/sources.list.d/ubuntu.sources ]; then \
      sed -i 's/^Components: .*/Components: main restricted universe multiverse/' /etc/apt/sources.list.d/ubuntu.sources; \
    fi; \
    rm -rf /var/lib/apt/lists/*

# System dependencies
RUN set -eux; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends --fix-missing \
      ca-certificates curl \
      git build-essential cmake pkg-config ninja-build \
      python3-venv python3-pip python3-dev \
      python3-requests python3-tqdm \
      libssl-dev libusb-1.0-0-dev libudev-dev \
      libgtk-3-dev libglfw3-dev libgl1-mesa-dev libglu1-mesa-dev \
      ffmpeg udev \
      python3-opencv \
    ; \
    rm -rf /var/lib/apt/lists/*

# Install ROS 2 Jazzy
RUN set -eux; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends --fix-missing \
      ros-jazzy-ros-base \
      ros-jazzy-cv-bridge \
      ros-jazzy-message-filters \
      ros-jazzy-image-transport \
      ros-jazzy-camera-info-manager \
      ros-jazzy-diagnostic-updater \
      ros-jazzy-launch \
      ros-jazzy-launch-ros \
      python3-rosdep \
      python3-colcon-common-extensions \
      python3-vcstool \
    ; \
    (rosdep init || true); \
    rosdep update; \
    rm -rf /var/lib/apt/lists/*

# Copy project
WORKDIR ${PROJECT_ROOT}
COPY . ${PROJECT_ROOT}

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:${PATH}

RUN python3 -m venv ${VIRTUAL_ENV} --system-site-packages && \
    ${VIRTUAL_ENV}/bin/python -m pip install -U pip setuptools wheel packaging && \
    \
    # Install your deps (may temporarily pull numpy 2.x)
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir \
      pycuda \
      ultralytics \
      onnx \
      onnxruntime \
      open-clip-torch \
      boxmot && \
    \
    (${VIRTUAL_ENV}/bin/pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true) && \
    \
    ${VIRTUAL_ENV}/bin/pip install --no-cache-dir --force-reinstall --no-deps numpy==1.26.4 && \
    \
    ${VIRTUAL_ENV}/bin/python -c "import numpy, cv2; print('numpy:', numpy.__version__, numpy.__file__); print('cv2:', cv2.__version__)"

# Dirs
RUN mkdir -p ${PROJECT_ROOT}/engine ${PROJECT_ROOT}/scripts ${PROJECT_ROOT}/launch && \
    chmod +x ${PROJECT_ROOT}/scripts/*.sh 2>/dev/null || true && \
    chmod +x ${PROJECT_ROOT}/src/*.py 2>/dev/null || true

# Entrypoint
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -e' \
  'source /opt/ros/jazzy/setup.bash' \
  'export PATH=/opt/venv/bin:$PATH' \
  'exec "$@"' \
  > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "/opt/person_following/launch/person_following.launch.py"]
