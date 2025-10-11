<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/poster-maskdino-train.jpg"/>  

# Train MaskDINO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](https://ecosystem.supervisely.com/apps/supervisely-ecosystem/MaskDINO/train)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/MaskDINO)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/maskdino/train.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/maskdino/train.png)](https://supervisely.com)

</div>

# Overview

Mask DINO extends DINO (DETR with Improved Denoising Anchor Boxes) by adding a mask prediction branch which supports all image segmentation tasks (instance, panoptic, and semantic). It makes use of the query embeddings from DINO to dotproduct a high-resolution pixel embedding map to predict a set of binary masks. Some key components in DINO are extended for segmentation through a shared architecture and training process. Mask DINO is simple, efficient, and scalable, and it can benefit from joint large-scale detection and segmentation datasets.

![maskdino architecture](https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_architecture.png)

# How To Run

**Step 0.** Run the app from context menu of the project with annotations or from the Ecosystem

**Step 1.** Select if you want to use cached project or redownload it

<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_train_0.png" width="100%" style='padding-top: 10px'>

**Step 2.** Select train / val split

<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_train_1.png" width="100%" style='padding-top: 10px'>

**Step 3.** Select the classes you want to train model on

<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_train_2.png" width="100%" style='padding-top: 10px'>

**Step 4.** Select the model you want to train

<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_train_3.png" width="100%" style='padding-top: 10px'>

**Step 5.** Configure hyperaparameters and select whether you want to use model evaluation and convert checkpoints to ONNX and TensorRT

<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_train_4.png" width="100%" style='padding-top: 10px'>

**Step 6.** Enter experiment name and start training

<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_train_5.png" width="100%" style='padding-top: 10px'>

**Step 7.** Monitor training progress

<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_train_6.png" width="100%" style='padding-top: 10px'>

# Acknowledgment

This app is based on the great work [MaskDINO](https://github.com/IDEA-Research/MaskDINO).