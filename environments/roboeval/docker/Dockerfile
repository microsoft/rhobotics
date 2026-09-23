FROM rho-training:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        libegl1 \
        libegl-dev \
        libgl1 \
        libglib2.0-0t64 \
        libgles2 \
        && rm -rf /var/lib/apt/lists/*

ENV MUJOCO_GL=egl \
    PYOPENGL_PLATFORM=egl \
    NVIDIA_DRIVER_CAPABILITIES=compute,graphics,utility,video \
    HF_HOME=/hf_home \
    ROBOEVAL_DATA_ROOT=/data \
    LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:${LD_LIBRARY_PATH}

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && \
    echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "libEGL_nvidia.so.0"}}' \
        > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

ARG ROBOEVAL_COMMIT=7199e5c4851bbf8eaf185d486735cbcb233f608a
RUN git config --global url."https://github.com/".insteadOf git@github.com: && \
    git clone --filter=blob:none https://github.com/Robo-Eval/RoboEval.git /opt/roboeval && \
    cd /opt/roboeval && \
    git checkout "${ROBOEVAL_COMMIT}" && \
    git submodule update --init --recursive

# Keep the versions of NumPy, Gymnasium, and safetensors supplied by the Rho
# image instead of applying RoboEval's older global pins.
RUN pip install --no-cache-dir \
        "backoff>=2.1.0" \
        "diffusers==0.38.0" \
        "dm_control==1.0.31" \
        "h5py>=3.10.0" \
        "hydra-core==1.3.*" \
        "mojo @ git+https://github.com/helen9975/mojo.git" \
        "mujoco==3.3.3" \
        mujoco_utils \
        "numba>=0.59.0" \
        opencv-python-headless \
        "pymunk==6.11.1" \
        "pynput==1.7.7" \
        pyquaternion \
        "pyzmq>=26.2.1" \
        pyyaml \
        wget \
        "zarr>=2.17.0" && \
    pip install --no-deps -e /opt/roboeval

COPY environments/roboeval/ /workspace/environments/roboeval/

RUN python -c "import roboeval; from roboeval.envs.lift_pot import LiftPotPositionAndOrientation"

WORKDIR /workspace

CMD ["/bin/bash"]
