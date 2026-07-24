import os
import ctypes

ctypes.windll.shcore.SetProcessDpiAwareness(1)

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")

_nvidia_dirs = [
    os.path.join(VENV_DIR, "Lib", "site-packages", "nvidia", "cublas", "bin"),
    os.path.join(VENV_DIR, "Lib", "site-packages", "nvidia", "cudnn", "bin"),
]
for _d in _nvidia_dirs:
    if os.path.isdir(_d) and _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

if __name__ == "__main__":
    import ui
    app = ui.SotvoxApp()
    app.run()
