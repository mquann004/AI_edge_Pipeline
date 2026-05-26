from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO


def run_detection(image_path: str, output_dir: str, model_name: str = "yolov8n.pt"):
    image_path = Path(image_path)
    output_dir = Path(output_dir)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(model_name)

    results = model(str(image_path))

    for index, result in enumerate(results):
        annotated_image = result.plot()
        output_path = output_dir / f"detection_result_{index}.jpg"

        success = cv2.imwrite(str(output_path), annotated_image)

        if not success:
            raise RuntimeError(f"Cannot save output image to {output_path}")

        print(f"Detection result saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run YOLO object detection on an image.")
    parser.add_argument("--image", type=str, default="images/dog.jpg", help="Path to input image")
    parser.add_argument("--output", type=str, default="outputs", help="Output folder")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model name or path")

    args = parser.parse_args()

    run_detection(
        image_path=args.image,
        output_dir=args.output,
        model_name=args.model
    )


if __name__ == "__main__":
    main()