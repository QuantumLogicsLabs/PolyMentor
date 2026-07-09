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

CHECKPOINT_DIR="${CHECKPOINT_DIR:-models_saved/polymentor-chatbot-lora}"
EVAL_PROMPTS="${EVAL_PROMPTS:-data/eval/eval_prompts.json}"

echo "PolyMentor evaluation — LoRA checkpoint vs Groq (blind LLM-judge)"
echo "Python:      $PYTHON"
echo "Checkpoint:  $CHECKPOINT_DIR"
echo ""

"$PYTHON" - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    raise SystemExit(f"PyTorch is not installed correctly: {exc}")

missing = []
for package in ("transformers", "peft"):
    try:
        __import__(package)
    except Exception:
        missing.append(package)

if missing:
    print("Missing packages:", ", ".join(missing))
    print()
    print("Install them in the active training venv:")
    print("  python -m pip install -r requirements-train.txt")
    sys.exit(1)

if not torch.cuda.is_available():
    print("CUDA is not available in this environment — generation will run on CPU")
    print("and will be slow. Use the venv you trained with for a reasonable runtime.")
PY

echo ""
"$PYTHON" -u src/evaluation/eval_chatbot.py --checkpoint-dir "$CHECKPOINT_DIR" --eval-prompts "$EVAL_PROMPTS"

echo ""
echo "Also worth doing manually before trusting a promotion:"
echo "  1. Ask PolyMentor to teach a concept and read the answer yourself."
echo "  2. Paste buggy code and confirm the suggested fix actually runs."
echo "  3. Spot-check a couple of the eval_chatbot.py verdicts in experiments/logs/."
