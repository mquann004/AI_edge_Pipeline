from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import datetime, timezone


def calculate_sha256(file_path: Path) -> str:
    sha256_hash = hashlib.sha256()

    with file_path.open("rb") as file:
        for byte_block in iter(lambda: file.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def create_model_package(
    model_path: str,
    output_dir: str,
    model_name: str,
    version: str,
    model_format: str = "onnx",
    image_size: int = 640
):
    model_path = Path(model_path)
    output_dir = Path(output_dir)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    package_dir = output_dir / f"{model_name}_{version}"
    package_dir.mkdir(parents=True, exist_ok=True)

    packaged_model_name = f"{model_name}.{model_format}"
    packaged_model_path = package_dir / packaged_model_name

    shutil.copy2(model_path, packaged_model_path)

    checksum = calculate_sha256(packaged_model_path)
    file_size = packaged_model_path.stat().st_size
    created_at = datetime.now(timezone.utc).isoformat()

    metadata = {
        "model_name": model_name,
        "version": version,
        "format": model_format,
        "image_size": image_size,
        "file_name": packaged_model_name,
        "file_size_bytes": file_size,
        "sha256": checksum,
        "created_at": created_at,
        "description": "YOLO model exported to ONNX for Edge AI deployment"
    }

    manifest = {
        "latest_version": version,
        "model_name": model_name,
        "format": model_format,
        "file_name": packaged_model_name,
        "file_size_bytes": file_size,
        "sha256": checksum,
        "created_at": created_at,
        "download_url": "TO_BE_UPDATED_IN_RELEASE_STAGE"
    }

    metadata_path = package_dir / "metadata.json"
    manifest_path = package_dir / "manifest.json"

    metadata_path.write_text(json.dumps(metadata, indent=4), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")

    zip_path = output_dir / f"{model_name}_{version}.zip"

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(packaged_model_path, arcname=packaged_model_name)
        zip_file.write(metadata_path, arcname="metadata.json")
        zip_file.write(manifest_path, arcname="manifest.json")

    print(f"Model package created successfully: {zip_path}")
    print(f"Metadata file: {metadata_path}")
    print(f"Manifest file: {manifest_path}")
    print(f"SHA256: {checksum}")


def main():
    parser = argparse.ArgumentParser(description="Create model metadata, manifest, and zip package.")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--output", type=str, default="packages", help="Output package folder")
    parser.add_argument("--name", type=str, default="yolov8n_edge_model", help="Model name")
    parser.add_argument("--version", type=str, default="v0.1.0", help="Model version")
    parser.add_argument("--format", type=str, default="onnx", help="Model format")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")

    args = parser.parse_args()

    create_model_package(
        model_path=args.model,
        output_dir=args.output,
        model_name=args.name,
        version=args.version,
        model_format=args.format,
        image_size=args.imgsz
    )


if __name__ == "__main__":
    main()