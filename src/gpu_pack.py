import ctypes
import json
import os
import re
import subprocess
import urllib.request
import zipfile

from constants import CUDA_DIR

PYPI_JSON_URL = "https://pypi.org/pypi/{package}/json"
WINDOWS_WHEEL_TAG = "win_amd64"
DOWNLOAD_CHUNK_BYTES = 1024 * 256
REQUEST_TIMEOUT_SECONDS = 60

REQUIRED_PACKAGES = (
    {"package": "nvidia-cublas-cu12", "major": None},
    {"package": "nvidia-cudnn-cu12", "major": 9},
)
REQUIRED_LIBRARIES = ("cublas64_12.dll", "cudnn64_9.dll")


def _parse_version(version_text):
    numbers = []
    for part in re.split(r"[.\-+]", version_text):
        if part.isdigit():
            numbers.append(int(part))
        else:
            break
    return tuple(numbers)


def is_installed():
    for library in REQUIRED_LIBRARIES:
        if not os.path.exists(os.path.join(CUDA_DIR, library)):
            return False
    return True


def libraries_available():
    if is_installed():
        return True
    else:
        for library in REQUIRED_LIBRARIES:
            try:
                ctypes.WinDLL(library)
            except OSError:
                return False
        return True


def detect_nvidia_gpu():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW)
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip().splitlines()[0].strip()
        else:
            return None
    except Exception:
        return None


def _resolve_wheel(package, major, log=None):
    url = PYPI_JSON_URL.format(package=package)
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        metadata = json.loads(response.read().decode("utf-8"))

    best_version = None
    best_wheel = None
    for version_text, files in metadata.get("releases", {}).items():
        version_numbers = _parse_version(version_text)
        if not version_numbers:
            continue
        if major is not None and version_numbers[0] != major:
            continue
        for file_info in files:
            filename = file_info.get("filename", "")
            if filename.endswith(".whl") and WINDOWS_WHEEL_TAG in filename and not file_info.get("yanked"):
                if best_version is None or version_numbers > best_version:
                    best_version = version_numbers
                    best_wheel = file_info

    if best_wheel is None:
        raise RuntimeError(f"No {WINDOWS_WHEEL_TAG} wheel found for {package}")

    if log:
        size_mb = (best_wheel.get("size") or 0) / (1024 * 1024)
        log(f"  {package}: {best_wheel['filename']} ({size_mb:.0f} MB)")
    return best_wheel


def _download(url, destination_path, on_progress=None, total_bytes=0, is_cancelled=None):
    downloaded_bytes = 0
    with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        if not total_bytes:
            total_bytes = int(response.headers.get("Content-Length") or 0)
        with open(destination_path, "wb") as output_file:
            while True:
                if is_cancelled and is_cancelled():
                    raise RuntimeError("Cancelled")
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output_file.write(chunk)
                downloaded_bytes += len(chunk)
                if on_progress:
                    on_progress(downloaded_bytes, total_bytes)
    return downloaded_bytes


def _extract_libraries(wheel_path, log=None):
    extracted_count = 0
    with zipfile.ZipFile(wheel_path) as archive:
        for member in archive.namelist():
            if member.lower().endswith(".dll") and "/bin/" in member.lower():
                target_path = os.path.join(CUDA_DIR, os.path.basename(member))
                with archive.open(member) as source, open(target_path, "wb") as target:
                    while True:
                        chunk = source.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        target.write(chunk)
                extracted_count += 1
    if log:
        log(f"  Extracted {extracted_count} library file(s)")
    return extracted_count


def install(on_status=None, on_progress=None, is_cancelled=None, log=None):
    os.makedirs(CUDA_DIR, exist_ok=True)
    temp_dir = os.path.join(CUDA_DIR, "_download")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        wheels = []
        for entry in REQUIRED_PACKAGES:
            if on_status:
                on_status(f"Locating {entry['package']}...")
            wheels.append((entry["package"], _resolve_wheel(entry["package"], entry["major"], log=log)))

        total_download_bytes = sum((wheel.get("size") or 0) for _, wheel in wheels)
        completed_bytes = 0

        for package, wheel in wheels:
            wheel_path = os.path.join(temp_dir, wheel["filename"])
            if on_status:
                on_status(f"Downloading {package}...")

            def report(chunk_bytes, _total, base=completed_bytes):
                if on_progress and total_download_bytes:
                    on_progress(base + chunk_bytes, total_download_bytes)

            _download(wheel["url"], wheel_path, on_progress=report,
                      total_bytes=wheel.get("size") or 0, is_cancelled=is_cancelled)
            completed_bytes += wheel.get("size") or 0

            if on_status:
                on_status(f"Installing {package}...")
            _extract_libraries(wheel_path, log=log)
            os.remove(wheel_path)

        if not is_installed():
            raise RuntimeError("GPU libraries were downloaded but are incomplete")

        if on_status:
            on_status("GPU acceleration ready")
        return True
    finally:
        try:
            for leftover in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, leftover))
            os.rmdir(temp_dir)
        except OSError:
            pass
