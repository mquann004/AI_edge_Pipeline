from pathlib import Path
import argparse
import shutil
from ultralytics import YOLO


def export_to_onnx(model_name: str, output_path: str, image_size: int = 640):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading YOLO model: {model_name}")
    model = YOLO(model_name)

    print("Exporting model to ONNX format...")
    exported_path = model.export(
        format="onnx",
        imgsz=image_size,
        opset=12,
        simplify=False
    )

    exported_path = Path(exported_path)

    if not exported_path.exists():
        raise FileNotFoundError(f"Exported ONNX model not found: {exported_path}")

    shutil.move(str(exported_path), str(output_path))

    print(f"ONNX model exported successfully to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Export YOLO model to ONNX format.")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO .pt model path or name")
    parser.add_argument("--output", type=str, default="models/yolov8n.onnx", help="Output ONNX path")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for export")

    args = parser.parse_args()

    export_to_onnx(
        model_name=args.model,
        output_path=args.output,
        image_size=args.imgsz
    )


if __name__ == "__main__":
    main()