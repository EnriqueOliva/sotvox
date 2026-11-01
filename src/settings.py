import json
import os

from constants import APP_DATA_DIR

SETTINGS_PATH = os.path.join(APP_DATA_DIR, "settings.json")


def load_settings(path=None):
    path = path or SETTINGS_PATH
    try:
        with open(path, encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError):
        loaded = {}
    return loaded if isinstance(loaded, dict) else {}


def save_settings(values, path=None):
    path = path or SETTINGS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = path + ".tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(values, handle, indent=2, sort_keys=True)
    os.replace(temporary_path, path)
