# Edge AI Pipeline for Automated Model Deployment

## 1. Project Overview

This project builds an **Edge AI Pipeline** that supports automated model testing, optimization, packaging, release, remote update, rollback, and real-time inference on an Edge device such as Raspberry Pi.

The main idea of this project is not only to run an AI model on a camera, but to simulate a complete AI deployment workflow similar to a real-world production system.

The system can:

- Run YOLO object detection on images and camera input.
- Export YOLO model from PyTorch `.pt` format to ONNX `.onnx`.
- Automatically test the model using GitHub Actions.
- Package the model with `metadata.json` and `manifest.json`.
- Publish the model package to GitHub Release.
- Allow an Edge device to download the latest model package remotely.
- Verify the model using SHA256 checksum before activation.
- Activate the model into the Edge runtime environment.
- Reject corrupted model packages and keep the current model safe.
- Run inference using the latest active model on Raspberry Pi.

---

## 2. Project Purpose

The purpose of this project is to demonstrate a complete workflow for deploying an AI model to an Edge device.

Instead of manually copying a model file to the device, this project builds an automated pipeline:

```text
Code Update
    ↓
GitHub Actions CI
    ↓
Model Export to ONNX
    ↓
Model Package Creation
    ↓
GitHub Release
    ↓
Remote Update Agent
    ↓
OTA Model Update
    ↓
Edge Runtime Inference
