import os
import sys

APP_NAME = "Sotvox"
IS_FROZEN = getattr(sys, "frozen", False)


def _resolve_resource_dir():
    if IS_FROZEN:
        return getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve_app_data_dir():
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return os.path.join(local_app_data, APP_NAME)
    else:
        return os.path.join(os.path.expanduser("~"), APP_NAME)


RESOURCE_DIR = _resolve_resource_dir()
APP_DATA_DIR = _resolve_app_data_dir()

DEFAULT_OUTPUT_DIR = os.path.join(os.path.expanduser("~"), "Documents", "sotvox-transcripts")
LOG_DIR = os.path.join(APP_DATA_DIR, "logs")
CUDA_DIR = os.path.join(APP_DATA_DIR, "cuda")
SOUNDS_DIR = os.path.join(RESOURCE_DIR, "sounds")
ASSETS_DIR = os.path.join(RESOURCE_DIR, "assets")
ICON_PATH = os.path.join(ASSETS_DIR, "sotvox.ico")

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".ts", ".flv",
}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".ts", ".flv"}

LANG_MAP = {
    "Spanish": "es", "English": "en", "Portuguese": "pt", "French": "fr",
    "German": "de", "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Korean": "ko",
}
