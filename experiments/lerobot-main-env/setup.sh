#!/usr/bin/env sh
set -eu

ROOT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
VENV_DIR="$ROOT_DIR/.venv"
LEROBOT_DIR="$ROOT_DIR/lerobot"

echo "==> Isolated LeRobot main test environment"
echo "Root: $ROOT_DIR"

if ! command -v python3 >/dev/null 2>&1; then
  echo "error: python3 is required but was not found in PATH" >&2
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "error: git is required but was not found in PATH" >&2
  exit 1
fi

if ! command -v brew >/dev/null 2>&1; then
  echo "warning: Homebrew was not found. If git-lfs is missing, install it manually." >&2
fi

if [ ! -d "$VENV_DIR" ]; then
  echo "==> Creating virtual environment"
  python3 -m venv "$VENV_DIR"
else
  echo "==> Reusing existing virtual environment"
fi

# shellcheck disable=SC1091
. "$VENV_DIR/bin/activate"

echo "==> Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

if ! command -v git-lfs >/dev/null 2>&1; then
  if command -v brew >/dev/null 2>&1; then
    echo "==> Installing git-lfs via Homebrew"
    brew install git-lfs
  else
    echo "error: git-lfs is required and could not be auto-installed" >&2
    exit 1
  fi
fi

echo "==> Initializing git-lfs"
git lfs install

if [ ! -d "$LEROBOT_DIR/.git" ]; then
  echo "==> Cloning huggingface/lerobot"
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
else
  echo "==> Reusing existing lerobot checkout"
fi

cd "$LEROBOT_DIR"

echo "==> Installing LeRobot editable from current checkout"
python -m pip install -e .

echo "==> Ensuring Hugging Face Hub is available"
python -m pip install "huggingface-hub>=1.5.0"

echo
echo "==> Setup complete"
echo "Activate with:"
echo "  . \"$VENV_DIR/bin/activate\""
echo
echo "Sanity checks:"
echo "  python -c \"import lerobot; print(getattr(lerobot, '__file__', 'no file'))\""
echo "  python -c \"import lerobot; print(getattr(lerobot, '__version__', 'no __version__'))\""
echo
echo "Minimal model load test:"
echo "  python -c \"from lerobot.policies.act.modeling_act import ACTPolicy; p = ACTPolicy.from_pretrained('ThaJpo/hf_act_recordpolicy2'); print(type(p))\""
