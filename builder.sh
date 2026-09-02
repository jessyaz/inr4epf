#!/usr/bin/env bash
set -euo pipefail

### BEFORE PROD
#EPFTOOLBOX_REPO="https://github.com/jeslago/epftoolbox.git"

### FOR DEV
EPFTOOLBOX_REPO="git@github.com:jeslago/epftoolbox.git"

LIB_DIR="lib/epftoolbox"

echo "[1/4] dirs"
mkdir -p data datasets/raw datasets/processed outputs runs lib
touch datasets/raw/.gitkeep datasets/processed/.gitkeep

echo "[2/4] epftoolbox"
if [ -d "${LIB_DIR}/.git" ]; then
    git -C "${LIB_DIR}" pull --ff-only
else
    git clone "${EPFTOOLBOX_REPO}" "${LIB_DIR}"
fi

echo "[3/4] install"
if command -v uv &> /dev/null; then
    uv add --editable "${LIB_DIR}"
else
    pip install -e "${LIB_DIR}" --break-system-packages 2>/dev/null || pip install -e "${LIB_DIR}"
fi

echo "[4/4] tree"
if command -v tree &> /dev/null; then
    tree -L 3 -I '__pycache__|*.pyc|.git|outputs|runs' .
else
    find . -maxdepth 3 -not -path '*/.git*' -not -path '*__pycache__*' -not -path './outputs/*' -not -path './runs/*' | sort
fi

echo "done"