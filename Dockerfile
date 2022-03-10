# Start FROM NVIDIA PyTorch image https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.10-py3

# Set some environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES all
ENV NVIDIA_VISIBLE_DEVICES all
ENV PYTHONPATH="/workspace/FeicoesOleosas/http/:/workspace/FeicoesOleosas/py/:${PYTHONPATH}"

##################################################################
# You should modify this to match your CPU compute capability
ENV MAX_JOBS=2
##################################################################

##################################################################
# You should modify this to match your geographic area
# See https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
ENV TZ=America/Sao_Paulo
##################################################################

# Expose the HTTP server port
EXPOSE 8000

# Install linux packages
RUN apt update
RUN apt install -y zip htop screen libgl1-mesa-glx bash build-essential python3-opencv

# Copy the list of dependencies
COPY py/requirements.txt .

# Install Python dependencies
RUN python -m pip install --upgrade pip
RUN pip uninstall -y torch torchvision torchtext
RUN pip install --no-cache -r requirements.txt torch==1.10.2+cu113 torchvision==0.11.3+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Cleanup
RUN rm requirements.txt

# Change the workdir
RUN mkdir /workspace/FeicoesOleosas
WORKDIR /workspace/FeicoesOleosas