#!/bin/bash
# Builds Sotvox.app and Sotvox-macOS.dmg. Run setup/setup.sh first.
set -euo pipefail

INSTALLER_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$INSTALLER_DIR")"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
DIST_DIR="$INSTALLER_DIR/dist_app"
APP="$DIST_DIR/Sotvox.app"
DMG="$PROJECT_ROOT/Sotvox-macOS.dmg"

if [ ! -x "$PYTHON" ]; then
    echo "Dev environment not found at $PYTHON. Run setup/setup.sh first." >&2
    exit 1
fi

echo "[1/3] Freezing the application with PyInstaller..."
"$PYTHON" -m PyInstaller --noconfirm --clean \
    --distpath "$DIST_DIR" \
    --workpath "$INSTALLER_DIR/build" \
    "$INSTALLER_DIR/sotvox.spec"

if [ ! -x "$APP/Contents/MacOS/Sotvox" ]; then
    echo "Frozen app missing: $APP" >&2
    exit 1
fi

leaked=$(find "$APP" -iname "*cudnn*" -o -iname "*cublas*" | head -1)
if [ -n "$leaked" ]; then
    echo "CUDA libraries leaked into the bundle: $leaked" >&2
    exit 1
fi

echo "[2/3] Signing the app (ad-hoc)..."
codesign --force --deep --sign - "$APP"
codesign --verify --deep --strict "$APP"
echo "      App size: $(du -sh "$APP" | cut -f1)"

echo "[3/3] Building the disk image..."
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT
cp -R "$APP" "$STAGING/"
ln -s /Applications "$STAGING/Applications"
rm -f "$DMG"
hdiutil create -volname "Sotvox" -srcfolder "$STAGING" -ov -format UDZO "$DMG" > /dev/null

echo "Built: $DMG ($(du -h "$DMG" | cut -f1))"
