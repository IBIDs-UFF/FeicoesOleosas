ARG PYTORCH="1.10.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

#############################################################
# You should modify this to match your GPU compute capability
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
#############################################################

#############################################################
# You should modify this to match your CPU compute capability
ENV MAX_JOBS=2
#############################################################

ENV NVIDIA_DRIVER_CAPABILITIES all
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"

# Create a working directory and copy the codebase into the image
RUN mkdir wd
WORKDIR /wd

RUN mkdir http
COPY http/* ./http

RUN mkdir py
COPY py/* ./py

# Install dependencies
RUN apt-get update
RUN apt-get install -y \
    bash build-essential cmake ffmpeg git libopenblas-dev ninja-build openssh-server tmux ubuntu-restricted-extras wget xauth xterm
RUN apt-get clean
RUN rm -rf /var/lib/apt/lists/*

RUN pip install -r python ./lib/requirements.txt
