FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libmagic1 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
 
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

RUN pip3 install --no-cache-dir torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip3 install --no-cache-dir packaging==22.0

RUN pip3 install --no-cache-dir setuptools==69.5.1
RUN pip3 install --no-cache-dir imagecorruptions==1.1.2
RUN pip3 install --no-cache-dir ffmpeg-python==0.2.0
RUN pip3 install --no-cache-dir protobuf==3.20.3

RUN python3 -m pip install --no-cache-dir 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip3 install --no-cache-dir Pillow==9.5.0
RUN pip3 install --no-cache-dir timm==1.0.16
RUN python3 -m pip install --no-cache-dir supervisely==6.73.553
RUN pip3 install supervisely[model-benchmark]==6.73.553
RUN pip3 install supervisely[training]==6.73.553

RUN apt-get update && apt-get -y install curl

RUN mkdir -p /weights

RUN curl -L -o /weights/maskdino_semantic_ade20k.pth "https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_semantic_ade20k.pth"
RUN curl -L -o /weights/maskdino_semantic_cityscapes.pth "https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_semantic_cityscapes.pth"

RUN apt-get update && apt-get -y install wget
RUN apt-get update && apt -y install wireguard iproute2

RUN git clone -b 5090-support https://github.com/supervisely-ecosystem/MaskDINO.git

ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 8.9 9.0 12.0+PTX"
ENV CUDA_HOME="/usr/local/cuda-12.8"
ENV FORCE_CUDA="1"
WORKDIR /MaskDINO/maskdino/modeling/pixel_decoder/ops
RUN python3 setup.py build install
RUN pip3 install .

ENV PYTHONPATH=/MaskDINO

LABEL "role"="development"