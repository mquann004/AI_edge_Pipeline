from pathlib import Path
import argparse
import json
import time

import cv2
from ultralytics import YOLO


def read_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))


def load_current_model(device_dir: Path):
    version_file = device_dir / "current_version.json"

    if not version_file.exists():
        raise FileNotFoundError(
            f"Current version file not found: {version_file}. "
            "Please run OTA update agent first."
        )

    version_data = read_json(version_file)

    file_name = version_data.get("file_name")
    version = version_data.get("version")
    model_name = version_data.get("model_name")

    if not file_name:
        raise ValueError("current_version.json is missing file_name")

    model_path = device_dir / "current" / file_name

    if not model_path.exists():
        raise FileNotFoundError(f"Current model file not found: {model_path}")

    print(f"Loading current model: {model_path}")
    print(f"Model name: {model_name}")
    print(f"Model version: {version}")

    model = YOLO(str(model_path))

    return model, model_path, version_data


def run_inference(
    device_dir: str,
    image_path: str,
    output_path: str,
    metrics_path: str
):
    device_dir = Path(device_dir)
    image_path = Path(image_path)
    output_path = Path(output_path)
    metrics_path = Path(metrics_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    model, model_path, version_data = load_current_model(device_dir)

    print(f"Running inference on image: {image_path}")

    start_time = time.perf_counter()
    results = model(str(image_path))
    end_time = time.perf_counter()

    latency_ms = (end_time - start_time) * 1000

    result = results[0]
    annotated_image = result.plot()

    success = cv2.imwrite(str(output_path), annotated_image)

    if not success:
        raise RuntimeError(f"Cannot save output image to: {output_path}")

    detections = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = result.names[class_id]

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence
            })

    metrics = {
        "model_path": str(model_path),
        "model_name": version_data.get("model_name"),
        "model_version": version_data.get("version"),
        "format": version_data.get("format"),
        "image_path": str(image_path),
        "output_path": str(output_path),
        "latency_ms": round(latency_ms, 2),
        "num_detections": len(detections),
        "detections": detections
    }

    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print(f"Inference completed.")
    print(f"Latency: {latency_ms:.2f} ms")
    print(f"Number of detections: {len(detections)}")
    print(f"Output image saved to: {output_path}")
    print(f"Metrics saved to: {metrics_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Edge runtime inference service using the currently active model."
    )

    parser.add_argument(
        "--device-dir",
        type=str,
        default="edge_device",
        help="Simulated edge device storage folder"
    )

    parser.add_argument(
        "--image",
        type=str,
        default="images/dog.jpg",
        help="Input image path"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="outputs/edge_runtime_result.jpg",
        help="Output image path"
    )

    parser.add_argument(
        "--metrics",
        type=str,
        default="outputs/edge_runtime_metrics.json",
        help="Output metrics JSON path"
    )

    args = parser.parse_args()

    run_inference(
        device_dir=args.device_dir,
        image_path=args.image,
        output_path=args.output,
        metrics_path=args.metrics
    )


if __name__ == "__main__":
    main()