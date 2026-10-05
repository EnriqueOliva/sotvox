#!/bin/bash
# Double-click in Finder to run Sotvox from source on macOS (the counterpart of launch.vbs).
BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="$BASE_DIR/.venv/bin/python"

if [ ! -x "$PYTHON" ]; then
    echo "Sotvox needs a one-time setup before it can run"
    echo "(it installs Python and the required components, and needs internet)."
    read -r -p "Run the setup now? [y/N] " answer
    case "$answer" in
        [yY]*) "$BASE_DIR/setup/setup.sh" || exit 1 ;;
        *) exit 0 ;;
    esac
fi

cd "$BASE_DIR"
nohup "$PYTHON" "$BASE_DIR/src/main.py" > /dev/null 2>&1 &
