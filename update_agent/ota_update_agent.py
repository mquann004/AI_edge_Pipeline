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


def read_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))


def write_json(file_path: Path, data: dict):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, indent=4), encoding="utf-8")


def prepare_device_folders(device_dir: Path):
    current_dir = device_dir / "current"
    previous_dir = device_dir / "previous"
    staging_dir = device_dir / "staging"
    logs_dir = device_dir / "logs"

    current_dir.mkdir(parents=True, exist_ok=True)
    previous_dir.mkdir(parents=True, exist_ok=True)
    staging_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, previous_dir, staging_dir, logs_dir


def log_event(logs_dir: Path, message: str):
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "ota_update.log"
    timestamp = datetime.now(timezone.utc).isoformat()
    line = f"[{timestamp}] {message}\n"

    with log_file.open("a", encoding="utf-8") as file:
        file.write(line)

    print(message)


def clear_folder(folder: Path):
    if folder.exists():
        shutil.rmtree(folder)

    folder.mkdir(parents=True, exist_ok=True)


def copy_folder_content(source: Path, destination: Path):
    clear_folder(destination)

    for item in source.iterdir():
        target = destination / item.name

        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def extract_package(package_path: Path, staging_dir: Path):
    if not package_path.exists():
        raise FileNotFoundError(f"Model package not found: {package_path}")

    clear_folder(staging_dir)

    with zipfile.ZipFile(package_path, "r") as zip_file:
        zip_file.extractall(staging_dir)


def verify_model(staging_dir: Path, manifest: dict):
    file_name = manifest.get("file_name")
    expected_sha256 = manifest.get("sha256")

    if not file_name:
        raise ValueError("manifest.json is missing file_name")

    if not expected_sha256:
        raise ValueError("manifest.json is missing sha256")

    model_path = staging_dir / file_name

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found in staging: {model_path}")

    actual_sha256 = calculate_sha256(model_path)

    if actual_sha256 != expected_sha256:
        raise ValueError(
            "SHA256 verification failed. "
            f"Expected: {expected_sha256}, Actual: {actual_sha256}"
        )

    if model_path.stat().st_size <= 0:
        raise ValueError("Model file is empty")

    return model_path


def get_current_version(device_dir: Path) -> str:
    version_file = device_dir / "current_version.json"

    if not version_file.exists():
        return "none"

    data = read_json(version_file)

    return data.get("version", "none")


def save_current_version(device_dir: Path, manifest: dict):
    version_data = {
        "model_name": manifest.get("model_name"),
        "version": manifest.get("latest_version"),
        "format": manifest.get("format"),
        "file_name": manifest.get("file_name"),
        "sha256": manifest.get("sha256"),
        "activated_at": datetime.now(timezone.utc).isoformat()
    }

    write_json(device_dir / "current_version.json", version_data)


def rollback(current_dir: Path, previous_dir: Path, logs_dir: Path):
    if not previous_dir.exists() or not any(previous_dir.iterdir()):
        log_event(logs_dir, "Rollback skipped: no previous model available.")
        return

    clear_folder(current_dir)
    copy_folder_content(previous_dir, current_dir)

    log_event(logs_dir, "Rollback completed: restored previous model.")


def ota_update(package_path: str, device_dir: str, force: bool = False):
    package_path = Path(package_path)
    device_dir = Path(device_dir)

    current_dir, previous_dir, staging_dir, logs_dir = prepare_device_folders(device_dir)

    try:
        log_event(logs_dir, "Starting OTA update process...")

        extract_package(package_path, staging_dir)
        log_event(logs_dir, f"Package extracted to staging: {staging_dir}")

        manifest_path = staging_dir / "manifest.json"
        manifest = read_json(manifest_path)

        new_version = manifest.get("latest_version")
        current_version = get_current_version(device_dir)

        if not new_version:
            raise ValueError("manifest.json is missing latest_version")

        log_event(logs_dir, f"Current version: {current_version}")
        log_event(logs_dir, f"New version: {new_version}")

        if current_version == new_version and not force:
            log_event(logs_dir, "Update skipped: device already has the latest version.")
            return

        model_path = verify_model(staging_dir, manifest)
        log_event(logs_dir, f"SHA256 verification passed for: {model_path.name}")

        if any(current_dir.iterdir()):
            copy_folder_content(current_dir, previous_dir)
            log_event(logs_dir, "Current model backed up to previous folder.")
        else:
            log_event(logs_dir, "No current model found. Fresh install mode.")

        copy_folder_content(staging_dir, current_dir)
        save_current_version(device_dir, manifest)

        log_event(logs_dir, "New model activated successfully.")
        log_event(logs_dir, "OTA update process completed.")

    except Exception as error:
        log_event(logs_dir, f"OTA update failed: {error}")
        rollback(current_dir, previous_dir, logs_dir)
        raise


def main():
    parser = argparse.ArgumentParser(description="Simulated OTA update agent for Edge AI model deployment.")
    parser.add_argument("--package", type=str, required=True, help="Path to model package .zip")
    parser.add_argument("--device-dir", type=str, default="edge_device", help="Simulated edge device storage folder")
    parser.add_argument("--force", action="store_true", help="Force update even if version is the same")

    args = parser.parse_args()

    ota_update(
        package_path=args.package,
        device_dir=args.device_dir,
        force=args.force
    )


if __name__ == "__main__":
    main()