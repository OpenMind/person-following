
# Production Dockerfile for Person Following System (Jetson Thor + ROS 2 Humble)
FROM nvcr.io/nvidia/pytorch:24.12-py3

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=humble \
    PROJECT_ROOT=/opt/person_following \
    VIRTUAL_ENV=/opt/venv \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
    PATH=/opt/venv/bin:/usr/local/bin:$PATH


# Prefer the UCX/UCC that ships in the base image (HPC-X), then CUDA.
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/ucc/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    PATH=/usr/src/tensorrt/bin:/usr/local/cuda/bin:${VIRTUAL_ENV}/bin:${PATH}

# Make sure ld.so can also find HPC-X libs
RUN if [ -d /opt/hpcx/ucx/lib ] && [ -d /opt/hpcx/ucc/lib ]; then \
      printf '%s\n' /opt/hpcx/ucx/lib /opt/hpcx/ucc/lib > /etc/ld.so.conf.d/hpcx.conf && ldconfig; \
    fi

COPY --from=docker.io/astral/uv:latest /uv /uvx /usr/local/bin/

# System dependencies
RUN set -eux; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends --fix-missing \
      ca-certificates \
      curl \
      git build-essential \
      cmake \
      pkg-config \
      ninja-build \
      python3-venv \
      python3-dev \
      python3-requests \
      python3-tqdm \
      libssl-dev \
      libusb-1.0-0-dev \
      libudev-dev \
      libgtk-3-dev \
      libglfw3-dev \
      libgl1-mesa-dev \
      libglu1-mesa-dev \
      ffmpeg \
      udev \
      python3-opencv \
    ; \
    rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble
RUN set -eux; \
    rm -rf /var/lib/apt/lists/*; \
    curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
      http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list; \
    apt-get update -o Acquire::Retries=5; \
    apt-get install -y --no-install-recommends --fix-missing \
      ros-humble-ros-base \
      ros-humble-cv-bridge \
      ros-humble-message-filters \
      ros-humble-image-transport \
      ros-humble-camera-info-manager \
      ros-humble-diagnostic-updater \
      ros-humble-launch \
      ros-humble-launch-ros \
      ros-humble-rmw-cyclonedds-cpp \
      python3-rosdep \
      python3-colcon-common-extensions \
      python3-vcstool \
    ; \
    (rosdep init || true); \
    rosdep update; \
    rm -rf /var/lib/apt/lists/*

WORKDIR ${PROJECT_ROOT}

# Cache deps layer: copy only lock + pyproject first
# (If you don't have uv.lock, remove it from this COPY and drop --locked below.)
COPY pyproject.toml uv.lock ./

# Create venv WITH system site-packages, then install deps from pyproject
RUN uv venv "${VIRTUAL_ENV}" --python python3 --system-site-packages && \
    uv sync --locked --no-install-project --all-extras

# Now copy the rest of the project
COPY . ${PROJECT_ROOT}

# Build ROS2 packages (om_api, unitree_api)
RUN source /opt/ros/humble/setup.bash && \
    cd ${PROJECT_ROOT} && \
    colcon build --symlink-install --packages-select om_api unitree_api

# Dirs
RUN mkdir -p ${PROJECT_ROOT}/engine ${PROJECT_ROOT}/scripts ${PROJECT_ROOT}/launch && \
    chmod +x ${PROJECT_ROOT}/scripts/*.sh 2>/dev/null || true && \
    chmod +x ${PROJECT_ROOT}/src/*.py 2>/dev/null || true

# Entrypoint - source both ROS and colcon workspace
RUN printf '%s\n' \
  '#!/usr/bin/env bash' \
  'set -e' \
  'source /opt/ros/humble/setup.bash' \
  'source /opt/person_following/install/setup.bash' \
  'export PATH=/opt/venv/bin:$PATH' \
  'exec "$@"' \
  > /entrypoint.sh && chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "/opt/person_following/launch/person_following_launch.py"]
