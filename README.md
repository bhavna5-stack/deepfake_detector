# Deepfake Detector using EfficientNetV2-S

---
title: Deepfake Detector
emoji: 🤖
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: "5.34.2"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Deepfake Detector using EfficientNetV2-S

AI-powered Deepfake Detection System built with PyTorch, EfficientNetV2-S, and Gradio.

## Features

- Image Deepfake Detection
- Batch Prediction
- Video Prediction
- Webcam Detection
- Grad-CAM Visualization
- PDF Report Generation
- EfficientNetV2-S Model

## Overview

This project detects whether an image is REAL or FAKE using a deep learning model based on EfficientNetV2-S.

---

## Features

- GPU Training (CUDA)
- EfficientNetV2-S
- Batch Prediction
- Image Prediction
- Video Prediction
- Webcam Detection
- Gradio Web App
- Prediction History
- CSV Export
- Resume Training
- Checkpoint Saving

---

## Dataset

CIFAKE Dataset

Train Images : 100000

Test Images : 20000

Classes

- FAKE
- REAL

---

## Tech Stack

Python

PyTorch

Torchvision

OpenCV

Albumentations

Gradio

Scikit-learn

Pandas

NumPy

CUDA

---

## Installation

pip install -r requirements.txt

---

## Train

python train.py

---

## Evaluate

python evaluate.py

---

## Predict

python predict.py

---

## Batch Prediction

python batch_predict.py

---

## Video Prediction

python video_predict.py

---

## Webcam Detection

python webcam.py

---

## Web Application

https://huggingface.co/spaces/Tanujthakur-5/deepfake-detector

---

## Folder Structure

DeepfakeDetector/

config.py

dataset.py

model.py

train.py

predict.py

batch_predict.py

video_predict.py

webcam.py

app.py

evaluate.py

train/

test/

models/

outputs/

---

## Model

EfficientNetV2-S

Optimizer : AdamW

Loss : CrossEntropy

Scheduler : CosineAnnealingLR

Mixed Precision Training

---

## Author 
Bhavna Agarwal
