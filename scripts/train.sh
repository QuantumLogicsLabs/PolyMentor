#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    if [[ -x "$VIRTUAL_ENV/Scripts/python.exe" ]]; then
        PYTHON="$VIRTUAL_ENV/Scripts/python.exe"
    elif [[ -x "$VIRTUAL_ENV/bin/python" ]]; then
        PYTHON="$VIRTUAL_ENV/bin/python"
    elif command -v cygpath >/dev/null 2>&1; then
        VENV_UNIX="$(cygpath -u "$VIRTUAL_ENV" 2>/dev/null || true)"
        if [[ -n "$VENV_UNIX" && -x "$VENV_UNIX/Scripts/python.exe" ]]; then
            PYTHON="$VENV_UNIX/Scripts/python.exe"
        else
            PYTHON="${PYTHON:-python}"
        fi
    else
        PYTHON="${PYTHON:-python}"
    fi
elif [[ -x "$PROJECT_ROOT/venv312/Scripts/python.exe" ]]; then
    PYTHON="$PROJECT_ROOT/venv312/Scripts/python.exe"
elif [[ -x "$PROJECT_ROOT/venv312/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/venv312/bin/python"
elif [[ -x "$PROJECT_ROOT/venv/Scripts/python.exe" ]]; then
    PYTHON="$PROJECT_ROOT/venv/Scripts/python.exe"
elif [[ -x "$PROJECT_ROOT/venv/bin/python" ]]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python"
else
    PYTHON="${PYTHON:-python}"
fi

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-Coder-0.5B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-models_saved/polymentor-chatbot-lora}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
INCLUDE_VENDOR="${INCLUDE_VENDOR:-1}"
MAX_VENDOR_FILES="${MAX_VENDOR_FILES:-200}"

echo "PolyMentor local fine-tuning"
echo "Python:      $PYTHON"
echo "Base model:  $BASE_MODEL"
echo "Output:      $OUTPUT_DIR"
echo ""

"$PYTHON" - <<'PY'
import sys

if sys.version_info >= (3, 13):
    print(f"Python: {sys.version.split()[0]}")
    print()
    print("PyTorch CUDA wheels on Windows are not available for this Python version.")
    print("Use Python 3.12 for RTX/CUDA training.")
    print()
    print("Create a fresh Windows venv like this:")
    print("  deactivate  # if a venv is active")
    print("  py -3.12 -m venv venv312")
    print("  .\\venv312\\Scripts\\Activate.ps1")
    print("  python -m pip install --upgrade pip")
    print("  python -m pip install -e .")
    print("  python -m pip install -r requirements.txt")
    print("  python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio")
    print("  bash scripts/train.sh")
    sys.exit(1)

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch is not installed correctly: {exc}")

missing = []
for package in ("transformers", "peft", "accelerate"):
    try:
        __import__(package)
    except Exception:
        missing.append(package)

if missing:
    print("Missing training packages:", ", ".join(missing))
    print()
    print("Install them in the active training venv:")
    print("  python -m pip install -r requirements-train.txt")
    sys.exit(1)

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print()
    print("CUDA is not available in this environment.")
    print("Your current PyTorch build is probably CPU-only.")
    print("Install a CUDA PyTorch build, then run this script again.")
    print()
    print("Example for many Windows/NVIDIA setups:")
    print("  python -m pip uninstall -y torch torchvision torchaudio")
    print("  python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio")
    sys.exit(1)
PY

mkdir -p "$OUTPUT_DIR"
mkdir -p experiments/logs
LOG_FILE="experiments/logs/finetune_chatbot_$(date +%Y%m%d_%H%M%S).log"

ARGS=(
    --base-model "$BASE_MODEL"
    --output-dir "$OUTPUT_DIR"
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --learning-rate "$LEARNING_RATE"
    --max-length "$MAX_LENGTH"
    --max-vendor-files "$MAX_VENDOR_FILES"
)

if [[ "$INCLUDE_VENDOR" == "1" || "$INCLUDE_VENDOR" == "true" ]]; then
    ARGS+=(--include-vendor)
fi

echo "Log file:    $LOG_FILE"
echo ""

export HF_HUB_DISABLE_SYMLINKS_WARNING=1

set +e
"$PYTHON" -u src/training/finetune_chatbot.py "${ARGS[@]}" 2>&1 | tee "$LOG_FILE"
STATUS=${PIPESTATUS[0]}
set -e

if [[ "$STATUS" -ne 0 ]]; then
    echo ""
    echo "Training failed with exit code $STATUS."
    echo "See log file: $LOG_FILE"
    exit "$STATUS"
fi

echo ""
echo "Training finished."
echo "Checkpoint saved in: $OUTPUT_DIR"
echo ""
echo "Note: this is a local LoRA adapter checkpoint. Groq remains the default"
echo "runtime unless you add a local-model inference path."
