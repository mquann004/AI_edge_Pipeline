# Edge AI Pipeline

## 1. Project Overview

This project aims to build an Edge AI pipeline that supports automatic model testing, optimization, deployment, and update on edge devices such as Raspberry Pi.

The first stage focuses on running an AI object detection model using YOLO and OpenCV.

## 2. Current Features

- Run YOLOv8 model on a test image
- Run YOLOv8 model with webcam
- Prepare project structure for future CI/CD pipeline

## 3. Technologies

- Python
- OpenCV
- Ultralytics YOLO
- Git
- GitHub
- Future: Docker, GitHub Actions, ONNX, Raspberry Pi

## 4. Project Structure

```text
edge-ai-pipeline/
├── inference/
│   ├── test_yolo.py
│   └── test_camera.py
├── images/
├── models/
├── requirements.txt
├── README.md
└── .gitignore