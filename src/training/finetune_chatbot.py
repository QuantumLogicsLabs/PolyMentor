"""
Fine-tune a small local PolyMentor chatbot adapter.

This script is intentionally separate from the Groq runtime. Groq remains the
default production model, while this creates a local LoRA adapter checkpoint for
experiments on an NVIDIA GPU such as an RTX 4050.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


LANGUAGE_BY_SUFFIX = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".c": "c",
    ".h": "c/cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
}


def load_json_records(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []


def sample_to_training_text(sample: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    language = sample.get("language", "code")
    level = sample.get("difficulty") or sample.get("difficulty_level") or "beginner"
    code = sample.get("code", "").strip()
    errors = ", ".join(sample.get("error_types") or [sample.get("error_type", "unknown")])
    explanation = sample.get("explanation", "Explain the issue clearly and teach the concept.")
    concept = sample.get("concept_taught", "Programming concept")
    hints = sample.get("hint_steps", [])
    hints_text = "\n".join(f"- {hint}" for hint in hints)

    user = (
        f"I am a {level} learner. Review this {language} code, identify likely "
        f"bugs, explain the concept, and show how to fix my thinking.\n\n"
        f"```{language}\n{code}\n```"
    )
    assistant = (
        f"Likely bugs:\n- {errors}\n\n"
        f"Explanation:\n{explanation}\n\n"
        f"Lesson:\n{concept}\n\n"
        f"Next steps:\n{hints_text or '- Try a small test case and trace the code line by line.'}"
    )
    return render_chat(tokenizer, user, assistant)


def prompt_to_training_text(sample: dict[str, Any], tokenizer: AutoTokenizer) -> str:
    user = str(sample.get("userMessage", "")).strip()
    assistant = str(sample.get("assistantMessage", "")).strip()
    return render_chat(tokenizer, user, assistant)


def vendor_to_training_text(path: Path, tokenizer: AutoTokenizer, max_chars: int) -> str | None:
    language = LANGUAGE_BY_SUFFIX.get(path.suffix.lower())
    if not language:
        return None
    try:
        code = path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return None
    if len(code) < 80:
        return None
    code = code[:max_chars]
    user = (
        f"Teach me how to read this {language} code snippet. Explain the syntax "
        "patterns and what a learner should notice.\n\n"
        f"```{language}\n{code}\n```"
    )
    assistant = (
        "This is a reference code-reading example. Focus on the structure, "
        "the names, the control flow, and the language syntax. A good way to "
        "study it is to identify the inputs, the important declarations, the "
        "main operations, and any edge cases before changing anything."
    )
    return render_chat(tokenizer, user, assistant)


def render_chat(tokenizer: AutoTokenizer, user: str, assistant: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are PolyMentor, a coding tutor chatbot. Teach code, help "
                "identify likely bugs, explain fixes clearly, and avoid numeric "
                "quality scores."
            ),
        },
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False)
    return (
        "System: "
        + messages[0]["content"]
        + "\n\nUser: "
        + user
        + "\n\nAssistant: "
        + assistant
    )


def build_texts(args: argparse.Namespace, tokenizer: AutoTokenizer) -> list[str]:
    texts: list[str] = []
    for pattern in args.data_globs:
        for path in sorted(Path().glob(pattern)):
            for sample in load_json_records(path):
                if sample.get("userMessage") and sample.get("assistantMessage"):
                    texts.append(prompt_to_training_text(sample, tokenizer))
                elif sample.get("code"):
                    texts.append(sample_to_training_text(sample, tokenizer))

    if args.include_vendor:
        vendor_files = []
        for suffix in LANGUAGE_BY_SUFFIX:
            vendor_files.extend(Path("vendor").rglob(f"*{suffix}"))
        random.Random(args.seed).shuffle(vendor_files)
        for path in vendor_files[: args.max_vendor_files]:
            text = vendor_to_training_text(path, tokenizer, args.max_vendor_chars)
            if text:
                texts.append(text)

    random.Random(args.seed).shuffle(texts)
    return texts


class TextDataset(Dataset):
    def __init__(self, texts: list[str], tokenizer: Any, max_length: int) -> None:
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            self.texts[index],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune PolyMentor chatbot LoRA adapter.")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--output-dir", default="models_saved/polymentor-chatbot-lora")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-vendor", action="store_true")
    parser.add_argument("--max-vendor-files", type=int, default=200)
    parser.add_argument("--max-vendor-chars", type=int, default=2400)
    parser.add_argument(
        "--data-globs",
        nargs="+",
        default=[
            "data/raw/*.json",
            "data/raw_samples/*.json",
            "data/processed/*.json",
        ],
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available in this Python environment. Install a CUDA "
            "PyTorch build before training on the RTX 4050."
        )

    print(f"Python: {sys.version.split()[0]}", flush=True)
    print(f"Torch: {torch.__version__}", flush=True)
    device = torch.device("cuda")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(
        f"CUDA memory before model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )
    print(f"Base model: {args.base_model}", flush=True)

    print("Loading training libraries...", flush=True)
    from huggingface_hub import hf_hub_download
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
    )
    print("Training libraries loaded.", flush=True)

    print("Downloading/checking base model files...", flush=True)
    model_path = Path("models_saved") / "hf_cache" / args.base_model.replace("/", "--")
    model_path.mkdir(parents=True, exist_ok=True)
    required_files = [
        "config.json",
        "generation_config.json",
        "merges.txt",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
    ]
    for filename in required_files:
        print(f"Downloading/checking {filename}...", flush=True)
        hf_hub_download(
            repo_id=args.base_model,
            filename=filename,
            local_dir=model_path,
        )
    print(f"Base model files ready: {model_path}", flush=True)

    print("Loading tokenizer...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Building training texts...", flush=True)
    texts = build_texts(args, tokenizer)
    if not texts:
        raise SystemExit("No training examples found. Add JSON samples under data/.")
    print(f"Training examples: {len(texts)}", flush=True)

    train_dataset = TextDataset(texts, tokenizer, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    print("Loading base model...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model.to(device)
    print(
        f"CUDA memory after model load: {torch.cuda.memory_allocated() / 1024**3:.2f} GB",
        flush=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = max(1, math.ceil(len(train_loader) * args.epochs / args.grad_accum))
    print(f"Optimization steps: {total_steps}", flush=True)

    print("Starting training...", flush=True)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    epochs = max(1, math.ceil(args.epochs))

    for epoch in range(epochs):
        running_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum
            loss.backward()
            running_loss += loss.item() * args.grad_accum

            if step % args.grad_accum == 0 or step == len(train_loader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % 5 == 0 or global_step == 1:
                    avg_loss = running_loss / step
                    print(
                        f"epoch={epoch + 1}/{epochs} step={global_step}/{total_steps} loss={avg_loss:.4f}",
                        flush=True,
                    )

        epoch_dir = Path(args.output_dir) / f"checkpoint-epoch-{epoch + 1}"
        epoch_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)
        print(f"Saved epoch checkpoint to {epoch_dir}", flush=True)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    (output_dir / "training_manifest.json").write_text(
        json.dumps(
            {
                "base_model": args.base_model,
                "examples": len(texts),
                "include_vendor": args.include_vendor,
                "cuda_device": torch.cuda.get_device_name(0),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved PolyMentor LoRA checkpoint to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
