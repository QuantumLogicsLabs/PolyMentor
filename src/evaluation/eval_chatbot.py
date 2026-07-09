"""
Blind Groq-judged comparison between the local LoRA chatbot checkpoint and
the production Groq model, with local promotion when the checkpoint wins.

Generation for the LoRA side runs locally (meant for the same GPU session
used to train the checkpoint, e.g. via scripts/train.sh). The Groq baseline
and the judge both go through the Groq API using the same GROQ_API_KEY the
rest of PolyMentor already uses — no extra paid infrastructure.

Promotion here only means "this checkpoint is good enough to consider" — it
copies the adapter to models_saved/polymentor-chatbot-lora-promoted/. Actually
serving a promoted checkpoint in place of Groq still requires a separate
GPU-hosted inference path, which is a deliberate future step, not automated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import PolyMentorPipeline  # noqa: E402

DEFAULT_EVAL_PROMPTS = PROJECT_ROOT / "data" / "eval" / "eval_prompts.json"
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / "models_saved" / "polymentor-chatbot-lora"
DEFAULT_PROMOTED_DIR = PROJECT_ROOT / "models_saved" / "polymentor-chatbot-lora-promoted"

SYSTEM_PROMPT = (
    "You are PolyMentor, a coding tutor chatbot. Teach code, help identify "
    "likely bugs, explain fixes clearly, and avoid numeric quality scores."
)

JUDGE_SYSTEM_PROMPT = (
    "You are grading two anonymized answers from different coding-tutor "
    "chatbots that responded to the same learner question. Judge strictly "
    "on teaching quality: technical correctness, clarity for the stated "
    "learner level, and whether the answer actually helps the learner "
    "understand and fix the problem. Respond with ONLY a JSON object of the "
    'form {"winner": "A"|"B"|"tie", "reason": "<one short sentence>"}.'
)

JUDGE_MODEL_DEFAULT = "llama-3.1-8b-instant"


@dataclass
class Verdict:
    prompt_index: int
    user_message: str
    winner: str  # "lora" | "groq" | "tie"
    reason: str


def load_eval_prompts(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if not prompts:
        raise SystemExit(f"No eval prompts found in {path}")
    return prompts


def generate_lora_answers(
    checkpoint_dir: Path,
    prompts: list[dict[str, Any]],
    max_new_tokens: int,
) -> list[str]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest_path = checkpoint_dir / "training_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(
            f"No training_manifest.json in {checkpoint_dir}; run scripts/train.sh first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_model_id = manifest["base_model"]
    base_model_path = PROJECT_ROOT / "models_saved" / "hf_cache" / base_model_id.replace("/", "--")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading base model from {base_model_path} on {device}...", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.float16 if device.type == "cuda" else torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.to(device)
    model.eval()

    answers: list[str] = []
    for item in prompts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": item["userMessage"]},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        answers.append(tokenizer.decode(generated, skip_special_tokens=True).strip())

    del model, base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return answers


def generate_groq_answers(prompts: list[dict[str, Any]]) -> list[str]:
    pipeline = PolyMentorPipeline.from_groq()
    answers = []
    for item in prompts:
        response = pipeline.chat(
            message=item["userMessage"],
            language=item.get("language", "python"),
            level=item.get("level", "beginner"),
        )
        answers.append(response.answer)
    return answers


def judge_pair(
    client: Any,
    judge_model: str,
    item: dict[str, Any],
    answer_a: str,
    answer_b: str,
) -> tuple[str, str]:
    user_content = (
        f"Learner level: {item.get('level', 'beginner')}\n"
        f"Language: {item.get('language', 'unknown')}\n"
        f"Question:\n{item['userMessage']}\n\n"
        f"Answer A:\n{answer_a}\n\n"
        f"Answer B:\n{answer_b}"
    )
    completion = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        max_completion_tokens=200,
    )
    content = completion.choices[0].message.content or ""
    try:
        start = content.index("{")
        end = content.rindex("}") + 1
        payload = json.loads(content[start:end])
        winner = str(payload.get("winner", "tie")).strip().upper()
        reason = str(payload.get("reason", "")).strip() or "no reason given"
        if winner not in {"A", "B"}:
            winner = "TIE"
        return ("tie" if winner == "TIE" else winner), reason
    except Exception:
        return "tie", f"unparseable judge output: {content[:120]!r}"


def run_eval(
    checkpoint_dir: Path,
    eval_path: Path,
    max_new_tokens: int,
    threshold: float,
    min_comparisons: int,
    seed: int,
) -> tuple[dict[str, Any], bool]:
    from groq import Groq

    prompts = load_eval_prompts(eval_path)

    print(f"Generating {len(prompts)} answers from local LoRA checkpoint...", flush=True)
    lora_answers = generate_lora_answers(checkpoint_dir, prompts, max_new_tokens)

    print(f"Generating {len(prompts)} baseline answers from Groq...", flush=True)
    groq_answers = generate_groq_answers(prompts)

    judge_model = os.getenv("GROQ_FALLBACK_MODEL", JUDGE_MODEL_DEFAULT)
    judge_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    rng = random.Random(seed)
    verdicts: list[Verdict] = []
    lora_wins = groq_wins = ties = 0

    print(f"Judging {len(prompts)} pairs with {judge_model} (blind, order-randomized)...", flush=True)
    for idx, item in enumerate(prompts):
        lora_first = rng.random() < 0.5
        answer_a, answer_b = (
            (lora_answers[idx], groq_answers[idx])
            if lora_first
            else (groq_answers[idx], lora_answers[idx])
        )
        raw_winner, reason = judge_pair(judge_client, judge_model, item, answer_a, answer_b)

        if raw_winner == "tie":
            winner = "tie"
            ties += 1
        else:
            picked_lora = (raw_winner == "A") == lora_first
            winner = "lora" if picked_lora else "groq"
            if winner == "lora":
                lora_wins += 1
            else:
                groq_wins += 1

        verdicts.append(Verdict(idx, item["userMessage"], winner, reason))
        print(f"  [{idx + 1}/{len(prompts)}] winner={winner} — {reason}", flush=True)

    non_ties = lora_wins + groq_wins
    win_rate = lora_wins / non_ties if non_ties else 0.0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint_dir": str(checkpoint_dir),
        "judge_model": judge_model,
        "eval_prompts_file": str(eval_path),
        "eval_prompts_hash": hashlib.sha256(eval_path.read_bytes()).hexdigest()[:12],
        "lora_wins": lora_wins,
        "groq_wins": groq_wins,
        "ties": ties,
        "non_tie_comparisons": non_ties,
        "lora_win_rate": win_rate,
        "promote_threshold": threshold,
        "promote_min_comparisons": min_comparisons,
        "verdicts": [asdict(v) for v in verdicts],
    }

    logs_dir = PROJECT_ROOT / "experiments" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    report_path = logs_dir / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("", flush=True)
    print("=" * 60, flush=True)
    print(f"LoRA wins: {lora_wins}   Groq wins: {groq_wins}   Ties: {ties}", flush=True)
    print(f"LoRA win rate (non-tie): {win_rate:.1%}", flush=True)
    print(f"Report saved to: {report_path}", flush=True)
    print("=" * 60, flush=True)

    promoted = False
    if non_ties >= min_comparisons and win_rate >= threshold:
        if DEFAULT_PROMOTED_DIR.exists():
            shutil.rmtree(DEFAULT_PROMOTED_DIR)
        shutil.copytree(checkpoint_dir, DEFAULT_PROMOTED_DIR)
        manifest = {
            "promoted_at": report["timestamp"],
            "source_checkpoint": str(checkpoint_dir),
            "lora_win_rate": win_rate,
            "lora_wins": lora_wins,
            "groq_wins": groq_wins,
            "ties": ties,
            "judge_model": judge_model,
            "eval_prompts_hash": report["eval_prompts_hash"],
        }
        (DEFAULT_PROMOTED_DIR / "promotion_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        promoted = True
        print(f"PROMOTED: checkpoint copied to {DEFAULT_PROMOTED_DIR}", flush=True)
        print(
            "Note: promotion only marks this checkpoint as promotion-worthy. "
            "Serving it in place of Groq still needs a separate GPU-hosted "
            "inference path (not automated) — Groq keeps serving /chat.",
            flush=True,
        )
    else:
        reason = (
            f"win rate {win_rate:.1%} below threshold {threshold:.1%}"
            if non_ties >= min_comparisons
            else f"only {non_ties} non-tie comparisons, need at least {min_comparisons}"
        )
        print(f"NOT PROMOTED: {reason}", flush=True)

    return report, promoted


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the local LoRA chatbot against Groq via blind LLM-judge, and promote locally if it wins."
    )
    parser.add_argument("--checkpoint-dir", default=str(DEFAULT_CHECKPOINT_DIR))
    parser.add_argument("--eval-prompts", default=str(DEFAULT_EVAL_PROMPTS))
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument(
        "--threshold", type=float, default=float(os.getenv("PROMOTE_THRESHOLD", "0.55"))
    )
    parser.add_argument(
        "--min-comparisons", type=int, default=int(os.getenv("PROMOTE_MIN_COMPARISONS", "10"))
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit(
            "GROQ_API_KEY is not set. It's required for the Groq baseline and the judge."
        )

    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise SystemExit(f"Checkpoint dir not found: {checkpoint_dir}. Run scripts/train.sh first.")

    eval_path = Path(args.eval_prompts)
    if not eval_path.exists():
        raise SystemExit(f"Eval prompts file not found: {eval_path}")

    run_eval(
        checkpoint_dir=checkpoint_dir,
        eval_path=eval_path,
        max_new_tokens=args.max_new_tokens,
        threshold=args.threshold,
        min_comparisons=args.min_comparisons,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
