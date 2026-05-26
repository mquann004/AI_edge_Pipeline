from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


def run_onnx_detection(model_path: str, image_path: str, output_dir: str):
    model_path = Path(model_path)
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ONNX model: {model_path}")
    model = YOLO(str(model_path))

    print(f"Running ONNX inference on image: {image_path}")
    results = model(str(image_path))

    for index, result in enumerate(results):
        annotated_image = result.plot()
        output_path = output_dir / f"onnx_detection_result_{index}.jpg"

        success = cv2.imwrite(str(output_path), annotated_image)

        if not success:
            raise RuntimeError(f"Cannot save output image to {output_path}")

        print(f"ONNX detection result saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO ONNX detection on an image.")
    parser.add_argument("--model", type=str, default="models/yolov8n.onnx", help="Path to ONNX model")
    parser.add_argument("--image", type=str, default="images/dog.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs", help="Output folder")

    args = parser.parse_args()

    run_onnx_detection(
        model_path=args.model,
        image_path=args.image,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()