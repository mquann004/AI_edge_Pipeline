from pathlib import Path
import argparse
import json
import urllib.request
import urllib.error
from urllib.parse import urlparse

from ota_update_agent import ota_update


def download_file(url: str, output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading from: {url}")
    print(f"Saving to: {output_path}")

    try:
        urllib.request.urlretrieve(url, output_path)
    except urllib.error.URLError as error:
        raise RuntimeError(f"Download failed: {error}") from error

    if not output_path.exists() or output_path.stat().st_size <= 0:
        raise RuntimeError(f"Downloaded file is invalid: {output_path}")

    print(f"Download completed: {output_path}")


def read_json(file_path: Path) -> dict:
    if not file_path.exists():
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    return json.loads(file_path.read_text(encoding="utf-8"))


def get_file_name_from_url(url: str, default_name: str) -> str:
    parsed_url = urlparse(url)
    file_name = Path(parsed_url.path).name

    if not file_name:
        return default_name

    return file_name


def remote_update(manifest_url: str, device_dir: str, download_dir: str, force: bool = False):
    download_dir = Path(download_dir)
    device_dir = Path(device_dir)

    manifest_path = download_dir / "remote_manifest.json"

    download_file(manifest_url, manifest_path)

    manifest = read_json(manifest_path)

    package_url = manifest.get("download_url")
    latest_version = manifest.get("latest_version")

    if not package_url:
        raise ValueError("Remote manifest is missing download_url")

    if not latest_version:
        raise ValueError("Remote manifest is missing latest_version")

    package_file_name = get_file_name_from_url(
        package_url,
        default_name=f"model_package_{latest_version}.zip"
    )

    package_path = download_dir / package_file_name

    download_file(package_url, package_path)

    print("Starting local OTA update with downloaded package...")

    ota_update(
        package_path=str(package_path),
        device_dir=str(device_dir),
        force=force
    )

    print("Remote update completed successfully.")


def main():
    parser = argparse.ArgumentParser(
        description="Remote update agent for downloading model package from GitHub Release."
    )

    parser.add_argument(
        "--manifest-url",
        type=str,
        required=True,
        help="URL to remote manifest.json"
    )

    parser.add_argument(
        "--device-dir",
        type=str,
        default="edge_device",
        help="Simulated edge device storage folder"
    )

    parser.add_argument(
        "--download-dir",
        type=str,
        default="downloads",
        help="Folder for downloaded manifest and model package"
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force update even if version is the same"
    )

    args = parser.parse_args()

    remote_update(
        manifest_url=args.manifest_url,
        device_dir=args.device_dir,
        download_dir=args.download_dir,
        force=args.force
    )


if __name__ == "__main__":
    main()