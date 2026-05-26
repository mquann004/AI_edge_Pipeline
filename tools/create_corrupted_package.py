from pathlib import Path
import argparse
import json
import zipfile
import shutil
import tempfile


def create_corrupted_package(source_package: str, output_package: str, bad_version: str):
    source_package = Path(source_package)
    output_package = Path(output_package)

    if not source_package.exists():
        raise FileNotFoundError(f"Source package not found: {source_package}")

    output_package.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        with zipfile.ZipFile(source_package, "r") as zip_file:
            zip_file.extractall(temp_dir)

        manifest_path = temp_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError("manifest.json not found in package")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        model_file_name = manifest.get("file_name")

        if not model_file_name:
            raise ValueError("manifest.json is missing file_name")

        model_path = temp_dir / model_file_name

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found in package: {model_path}")

        # Đổi version để OTA Agent xem đây là model mới
        manifest["latest_version"] = bad_version

        # Cố tình làm hỏng model nhưng KHÔNG cập nhật SHA256
        # Vì SHA256 trong manifest vẫn là của file cũ, bước verify sẽ fail
        with model_path.open("ab") as file:
            file.write(b"\nCORRUPTED_MODEL_CONTENT")

        manifest_path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")

        with zipfile.ZipFile(output_package, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for item in temp_dir.iterdir():
                zip_file.write(item, arcname=item.name)

    print(f"Corrupted package created: {output_package}")


def main():
    parser = argparse.ArgumentParser(description="Create a corrupted model package for rollback testing.")
    parser.add_argument("--source-package", type=str, required=True, help="Valid source model package")
    parser.add_argument("--output", type=str, default="packages/corrupted_model_package.zip", help="Output corrupted package")
    parser.add_argument("--bad-version", type=str, default="v0.1.bad", help="Fake bad version")

    args = parser.parse_args()

    create_corrupted_package(
        source_package=args.source_package,
        output_package=args.output,
        bad_version=args.bad_version
    )


if __name__ == "__main__":
    main()