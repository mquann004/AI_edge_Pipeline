import cv2
import numpy as np
from ultralytics import YOLO


def test_opencv_import():
    assert cv2.__version__ is not None


def test_numpy_import():
    assert np.__version__ is not None


def test_yolo_model_load():
    model = YOLO("yolov8n.pt")
    assert model is not None