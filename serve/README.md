<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/poster-maskdino-serve.jpg"/>  

# Serve MaskDINO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../supervisely-ecosystem/MaskDINO/serve)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/MaskDINO)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/maskdino/serve.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/maskdino/serve.png)](https://supervisely.com)

</div>

# Overview

Mask DINO extends DINO (DETR with Improved Denoising Anchor Boxes) by adding a mask prediction branch which supports all image segmentation tasks (instance, panoptic, and semantic). It makes use of the query embeddings from DINO to dotproduct a high-resolution pixel embedding map to predict a set of binary masks. Some key components in DINO are extended for segmentation through a shared architecture and training process. Mask DINO is simple, efficient, and scalable, and it can benefit from joint large-scale detection and segmentation datasets.

![maskdino architecture](https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_architecture.png)

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_serve_0.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/MaskDINO/releases/download/v0.0.1/maskdino_serve_1.png)

# Acknowledgment

This app is based on the great work [MaskDINO](https://github.com/IDEA-Research/MaskDINO).