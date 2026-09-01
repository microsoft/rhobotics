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
ENV LIBERO_CONFIG_PATH="/opt/libero-config/"
ENV LD_LIBRARY_PATH="/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}"

#requirements for headless GPU rendering, see also https://github.com/mmatl/pyrender/issues/149
RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
        echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

COPY environments/libero/ /workspace/environments/libero/

ENV CMAKE_POLICY_VERSION_MINIMUM=3.5
RUN pip install libero --use-pep517 --no-build-isolation

# libero and robosuite both depend on mujoco unpinned; robosuite 1.4.0 reads MjData.qM,
# which mujoco renamed to M in 3.4. Pin below the rename.
RUN pip install "mujoco==3.3.6" --no-cache-dir

# Install robosuite's default private macros file so imports do not emit the
# optional user-macros warning.
RUN python -c "import importlib.util, pathlib, shutil; \
p = pathlib.Path(next(iter(importlib.util.find_spec('robosuite').submodule_search_locations))); \
shutil.copyfile(p / 'macros.py', p / 'macros_private.py')"

# Pre-download libero assets from HuggingFace Hub into the pip-installed package
# directory so they are available offline at runtime (avoids HF rate limits).
# Then generate a config.yaml whose paths match the actual install location.
RUN python /workspace/environments/libero/docker/setup_libero_assets.py

# Remove unused system Python packages to avoid CVE warnings
# NOTE: Must be done after all apt operations to prevent reinstallation
RUN dpkg-query -W -f='${binary:Package} ${db:Status-Abbrev}\n' \
        python3-wheel python3-setuptools python3-jwt python3-cryptography \
        2>/dev/null | awk '$2 ~ /^ii/ {print $1}' | \
        xargs -r apt-get remove -y --allow-change-held-packages && \
    rm -rf /usr/lib/python3/dist-packages/*

# clean conda, pip, and huggingface download caches
RUN rm -rf ~/.cache/pip ~/.cache/huggingface

# Set working directory
WORKDIR /workspace

# Default command - keep container running for interactive use
CMD ["/bin/bash"]
