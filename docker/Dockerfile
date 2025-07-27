FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    wget \
    curl \
    git \
    ffmpeg \
    libgeos-dev \
    libsm6 \
    libexiv2-dev \
    libxrender-dev \
    libboost-all-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install --no-cache-dir packaging==22.0

RUN pip3 install setuptools==69.5.1
RUN pip3 install imagecorruptions==1.1.2
RUN pip3 install ffmpeg-python==0.2.0
RUN pip3 install protobuf==3.20.3

RUN python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN pip3 install Pillow==9.5.0
RUN pip3 install timm==1.0.16
RUN python3 -m pip install --no-cache-dir supervisely==6.73.404
RUN pip3 install supervisely[model-benchmark]==6.73.404
RUN pip3 install supervisely[training]==6.73.404

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN apt-get update && apt-get -y install curl

RUN mkdir -p /weights

RUN curl -L -o /weights/maskdino_semantic_ade20k.pth "https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_semantic_ade20k.pth"
RUN curl -L -o /weights/maskdino_semantic_cityscapes.pth "https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_semantic_cityscapes.pth"

LABEL python_sdk_version=6.73.404