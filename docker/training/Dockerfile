FROM nvidia/cuda:12.6.3-devel-ubuntu24.04

ENV PYTHONUNBUFFERED=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        TZ=Etc/UTC \
        DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get upgrade -y && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa -y
RUN apt-get install -y git

ENV MINICONDA_VERSION=py310_23.10.0-1
ENV PATH=/opt/miniconda/bin:$PATH
RUN apt-get update && \
        apt-get install -y --no-install-recommends wget runit

RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh && \
        bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
        conda update --all -c conda-forge -y && \
        conda clean -ay && \
        rm -rf /opt/miniconda/pkgs && \
        rm /tmp/miniconda.sh && \
        find / -type d -name __pycache__ | xargs rm -rf

ENV AZUREML_CONDA_ENVIRONMENT_PATH=/azureml-envs/default

# Create conda environment with py310
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
        python=3.10 \
        -c conda-forge

ENV PATH=$AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

ENV CONDA_DEFAULT_ENV=$AZUREML_CONDA_ENVIRONMENT_PATH

ENV CONDA_PREFIX=$AZUREML_CONDA_ENVIRONMENT_PATH

# Bootstrap pip inside the azureml conda env so that `pip` and `python`
# resolve to the same environment (miniconda's pip would install into
# /opt/miniconda instead of /azureml-envs/default).
RUN python -m ensurepip && python -m pip install --upgrade pip

COPY pyproject.toml /workspace/
COPY rho/ /workspace/rho/
COPY rho_client/ /workspace/rho_client/

WORKDIR /workspace
RUN pip install -e ".[server]"

RUN conda install -y -c conda-forge ffmpeg==7.1.1

# torch installation
#RUN pip install --no-cache-dir torch==2.8.0 torchvision==0.23.0
#RUN pip install torchcodec==0.6.0
RUN pip install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

# Upgrade wheel to fix CVE-2026-24049, wandb for Go stdlib CVEs
RUN pip install --upgrade wheel>=0.46.2 wandb>=0.24.0

# Install InfiniBand/RDMA libraries for multi-node training
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
        libibverbs1 \
        libibverbs-dev \
        ibverbs-providers \
        rdma-core \
        ibverbs-utils \
        librdmacm1 \
        libibumad3 \
        && rm -rf /var/lib/apt/lists/*

# Install HPC-X for NCCL RDMA support
ENV HPCX_VERSION=v2.20
RUN cd /tmp && \
        wget -q https://content.mellanox.com/hpc/hpc-x/${HPCX_VERSION}/hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu24.04-cuda12-x86_64.tbz && \
        tar -xjf hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu24.04-cuda12-x86_64.tbz -C /opt && \
        mv /opt/hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu24.04-cuda12-x86_64 /opt/hpcx && \
        rm -f hpcx-${HPCX_VERSION}-gcc-mlnx_ofed-ubuntu24.04-cuda12-x86_64.tbz

# Set LD_LIBRARY_PATH for InfiniBand/NCCL RDMA
ENV LD_LIBRARY_PATH=/opt/hpcx/ucx/lib:/opt/hpcx/sharp/lib:/opt/hpcx/nccl_rdma_sharp_plugin/lib:${LD_LIBRARY_PATH}

# Remove unused system Python packages to avoid CVE warnings
RUN apt-get remove -y python3-wheel python3-setuptools python3-jwt python3-cryptography || true

# clean conda and pip caches
RUN rm -rf ~/.cache/pip

# Configure bash for better interactive experience
RUN echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /root/.bashrc && \
        echo 'alias ll="ls -alF"' >> /root/.bashrc && \
        echo 'alias la="ls -A"' >> /root/.bashrc && \
        echo 'alias l="ls -CF"' >> /root/.bashrc && \
        echo 'alias ..="cd .."' >> /root/.bashrc && \
        echo 'alias ...="cd ../.."' >> /root/.bashrc && \
        echo 'export PYTHONPATH="/workspace:$PYTHONPATH"' >> /root/.bashrc

# Set working directory
WORKDIR /workspace

# Default command - keep container running for interactive use
CMD ["/bin/bash"]
