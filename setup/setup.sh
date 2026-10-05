#!/bin/bash
# Sotvox developer environment for macOS (the counterpart of setup.ps1).
# End users do not need this - they install Sotvox.app from Sotvox-macOS.dmg.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

step() { printf '\n  [%s] %s\n  %s\n' "$1" "$2" "--------------------------------------------------"; }

echo
echo "  ============================================="
echo "       Sotvox - developer environment (macOS)"
echo "  ============================================="
echo "  Project: $PROJECT_ROOT"

step "1/4" "Checking uv (Python manager)..."
export PATH="$HOME/.local/bin:$PATH"
if command -v uv > /dev/null; then
    echo "       Already installed: $(uv --version)"
else
    echo "       Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    echo "       Installed: $(uv --version)"
fi

step "2/4" "Installing Python 3.11 via uv..."
uv python install 3.11

step "3/4" "Creating virtual environment..."
if [ -d .venv ]; then
    echo "       Virtual environment already exists. Recreating..."
    rm -rf .venv
fi
uv venv --python 3.11

step "4/4" "Installing dependencies (this may take a few minutes)..."
uv pip install faster-whisper tkinterdnd2 pyinstaller pytest

echo
echo "  ============================================="
echo "       Developer setup complete"
echo "  ============================================="
echo
echo "  Run the app from source:"
echo "       ./launch.command   (or double-click it in Finder)"
echo "  Run the tests:"
echo "       .venv/bin/python -m pytest tests"
echo "  Build Sotvox.app and the disk image:"
echo "       installer/build_mac.sh"
echo
