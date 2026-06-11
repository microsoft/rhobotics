FROM rho-training:latest


# Install system dependencies for headless GPU rendering
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        libegl1 \
        libgles2 \
        libgl1 \
        libglib2.0-0t64 \
        libegl-dev \
        && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        nano \
        vim \
        build-essential \
        cmake \
        curl \
        wget \
        htop \
        tree \
        less \
        tmux \
        screen \
        bash-completion \
        sudo \
        nvtop \
        && rm -rf /var/lib/apt/lists/*

# Configure headless GPU rendering environment variables
ENV MUJOCO_GL="egl"
ENV PYOPENGL_PLATFORM="egl"
ENV NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video
# ENV LIBGL_ALWAYS_SOFTWARE=1
ENV HF_HOME="/hf_home"
ENV LIBERO_CONFIG_PATH="/workspace/.libero/"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

#requirements for headless GPU rendering, see also https://github.com/mmatl/pyrender/issues/149
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
        echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY environments/ /workspace/environments/

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN pip install libero --use-pep517 --no-build-isolation

# Need to downgrade numpy for libero
RUN pip install numpy==1.26.4

# Pre-download libero assets from HuggingFace Hub into the pip-installed package
# directory so they are available offline at runtime (avoids HF rate limits).
# Then generate a config.yaml whose paths match the actual install location.
RUN python /workspace/environments/libero/docker/setup_libero_assets.py

# Remove unused system Python packages to avoid CVE warnings
RUN apt-get remove -y python3-wheel python3-setuptools python3-jwt python3-cryptography || true

# clean conda, pip, and huggingface download caches
RUN rm -rf ~/.cache/pip ~/.cache/huggingface

# Set working directory
WORKDIR /workspace

# Default command - keep container running for interactive use
CMD ["/bin/bash"]
