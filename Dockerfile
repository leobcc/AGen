# --------------------------------------------------------------------------------------------------------------------
# The image needs to be run with: $ docker run -it --gpus all --name AGen -v /data-net/hupba/lbocchi:/AGen/data agen_image --rm -p 8763:8888 --shm-size 10gb --name lb
# If changes are not being included in the build use $ docker build --no-cache -t agen_image . 
# To re-enter the running container: $ docker exec -it <container_name_or_id> /bin/bash
# To start the container if it is already up but it is not running: $ docker start <container_name_or_id>
# To attach to a running container with a terminal and resume the process: $ docker attach <container_name_or_id>
# To select the gpu to be used: $ export CUDA_VISIBLE_DEVICES=1,2
# To copy the output folder: $ docker cp ec8cafda7c44:/IF3D/outputs/Video/parkinglot /home-net/lbocchi/IF3D_Project
# --------------------------------------------------------------------------------------------------------------------

# Build the container using the CUDA toolkit version 11.1 image as the base image
FROM nvidia/cuda:11.1.1-devel-ubuntu20.04

# Fetch the keys that are missing from the cuda base image 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

# For more details on the installation refer to https://saturncloud.io/blog/how-to-use-gpus-inside-docker-containers-resolving-cuda-version-and-torchcudaisavailable-issues/
# Also, refer to https://www.howtogeek.com/devops/how-to-use-an-nvidia-gpu-with-docker-containers/
RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-cudart-11-1 \
    && rm -rf /var/lib/apt/lists/*

# Install general dependecies
RUN apt-get update && \
    apt-get install -y \
    python3.7 \
    python3-pip \
    build-essential \
    git \
    git-lfs 

# Upgrade pip (this is necessary otherwise the installation of setuptools gives problems for some versions)
RUN python3 -m pip install --upgrade pip

# Downgrade numpy to  version 1.23.1 to avoid ImportError: cannot import name 'bool' from 'numpy'
RUN python3 -m pip install numpy==1.23.1

# Set environment variables
ENV FORCE_CUDA=1
ENV TORCH_CUDA_ARCH_LIST="6.1"

# Clone the IF3D repository. 
RUN git clone https://huggingface.co/spaces/leobcc/AGen

# Change working directory to the project folder
WORKDIR /AGen

# Install the cuda compatible version of torch 1.9.1
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# Install dependencies
RUN pip install -r requirements.txt

# Install PyTorch3D
RUN pip install git+https://github.com/facebookresearch/pytorch3d.git@stable

# Run setup
RUN cd VideoReconstructionModel/code; python3 setup.py develop; cd ..;

# Install Kaolin (version 0.10.0)
RUN git clone --recursive https://github.com/NVIDIAGameWorks/kaolin \
    && cd kaolin \
    && git checkout v0.10.0 \
    && python3 setup.py develop \
    && cd ..