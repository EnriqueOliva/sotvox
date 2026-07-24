import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
LOG_DIR = os.path.join(PROJECT_ROOT, "log")
SOUNDS_DIR = os.path.join(PROJECT_ROOT, "sounds")

SUPPORTED_EXTENSIONS = {
    ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".wma", ".aac",
    ".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".ts", ".flv",
}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".ts", ".flv"}

LANG_MAP = {
    "Spanish": "es", "English": "en", "Portuguese": "pt", "French": "fr",
    "German": "de", "Italian": "it", "Japanese": "ja", "Chinese": "zh", "Korean": "ko",
}

