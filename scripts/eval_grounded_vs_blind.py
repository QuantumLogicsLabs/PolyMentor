#!/usr/bin/env python3
"""
eval_grounded_vs_blind.py
-------------------------
Measure AdvancedCodeAnalyzer grounding on a fixed snippet fixture, and
optionally compare grounded vs blind Groq answer keyword coverage.

Usage:
  python scripts/eval_grounded_vs_blind.py
  python scripts/eval_grounded_vs_blind.py --with-groq
  python scripts/eval_grounded_vs_blind.py --fixture data/eval/grounded_chat_snippets.json --min-hit-rate 0.8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer  # noqa: E402

DEFAULT_FIXTURE = PROJECT_ROOT / "data" / "eval" / "grounded_chat_snippets.json"
DEFAULT_LOG_DIR = PROJECT_ROOT / "experiments" / "logs"


def load_fixture(path: Path) -> list[dict]:
    snippets = json.loads(path.read_text(encoding="utf-8"))
    if not snippets:
        raise SystemExit(f"Fixture is empty: {path}")
    return snippets


def pack_findings(analysis: dict, max_chars: int = 800) -> str:
    lines: list[str] = []
    for error in analysis.get("errors") or []:
        msg = error.get("message") or ""
        suggestion = error.get("suggestion") or ""
        lines.append(f"- [{error.get('category')}] {msg}")
        if suggestion:
            lines.append(f"  suggestion: {suggestion}")
    packed = "Static analysis findings:\n" + "\n".join(lines)
    if len(packed) > max_chars:
        packed = packed[: max_chars - 20] + "\n... [truncated]"
    return packed


def analyzer_hit(snippet: dict, analysis: dict) -> bool:
    expected_cats = set(snippet.get("expected_categories") or [])
    expected_keywords = [k.lower() for k in (snippet.get("expected_keywords") or [])]
    errors = analysis.get("errors") or []
    cats = {e.get("category") for e in errors}
    if expected_cats and cats & expected_cats:
        return True
    blob = " ".join(
        f"{e.get('message') or ''} {e.get('suggestion') or ''}" for e in errors
    ).lower()
    return any(keyword in blob for keyword in expected_keywords)


def keyword_coverage(text: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    lowered = (text or "").lower()
    hits = sum(1 for keyword in keywords if keyword.lower() in lowered)
    return hits / len(keywords)


def run_deterministic(snippets: list[dict]) -> dict:
    rows = []
    hits = 0
    for snippet in snippets:
        analysis = AdvancedCodeAnalyzer.analyze(snippet["code"], snippet["language"])
        ok = analyzer_hit(snippet, analysis)
        hits += int(ok)
        rows.append(
            {
                "id": snippet["id"],
                "hit": ok,
                "categories_found": sorted(
                    {e.get("category") for e in (analysis.get("errors") or [])}
                ),
                "messages": [e.get("message") for e in (analysis.get("errors") or [])],
            }
        )
        mark = "PASS" if ok else "MISS"
        print(f"  [{mark}] {snippet['id']}")
    hit_rate = hits / len(snippets)
    print(f"\nAnalyzer hit-rate: {hit_rate:.1%} ({hits}/{len(snippets)})")
    return {
        "mode": "deterministic",
        "hit_rate": hit_rate,
        "hits": hits,
        "total": len(snippets),
        "rows": rows,
    }


async def run_with_groq(snippets: list[dict]) -> dict:
    from src.inference.pipeline import PolyMentorPipeline

    pipeline = PolyMentorPipeline.from_groq()
    blind_scores: list[float] = []
    grounded_scores: list[float] = []
    rows = []

    for snippet in snippets:
        analysis = AdvancedCodeAnalyzer.analyze(snippet["code"], snippet["language"])
        question = snippet.get("question") or "What is wrong with this code?"
        keywords = snippet.get("expected_keywords") or []

        blind = await pipeline.analyze(
            code=snippet["code"],
            language=snippet["language"],
            level="beginner",
            question=question,
        )
        grounded_question = (
            f"{pack_findings(analysis)}\n\n"
            f"Using the findings above, answer the learner:\n{question}"
        )
        grounded = await pipeline.analyze(
            code=snippet["code"],
            language=snippet["language"],
            level="beginner",
            question=grounded_question,
        )

        blind_cov = keyword_coverage(blind.answer or "", keywords)
        grounded_cov = keyword_coverage(grounded.answer or "", keywords)
        blind_scores.append(blind_cov)
        grounded_scores.append(grounded_cov)

        rows.append(
            {
                "id": snippet["id"],
                "blind_coverage": blind_cov,
                "grounded_coverage": grounded_cov,
                "delta": grounded_cov - blind_cov,
                "blind_status": blind.status,
                "grounded_status": grounded.status,
            }
        )
        print(
            f"  {snippet['id']}: blind={blind_cov:.0%} grounded={grounded_cov:.0%} "
            f"delta={grounded_cov - blind_cov:+.0%}"
        )

    mean_blind = sum(blind_scores) / len(blind_scores)
    mean_grounded = sum(grounded_scores) / len(grounded_scores)
    delta = mean_grounded - mean_blind
    print(
        f"\nMean keyword coverage — blind={mean_blind:.1%} "
        f"grounded={mean_grounded:.1%} delta={delta:+.1%}"
    )
    return {
        "mode": "with_groq",
        "mean_blind_coverage": mean_blind,
        "mean_grounded_coverage": mean_grounded,
        "delta": delta,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate grounded analyzer coverage vs blind Groq."
    )
    parser.add_argument(
        "--fixture",
        default=str(DEFAULT_FIXTURE),
        help="Path to grounded_chat_snippets.json",
    )
    parser.add_argument(
        "--min-hit-rate",
        type=float,
        default=0.8,
        help="Minimum analyzer hit-rate required to pass (default 0.8)",
    )
    parser.add_argument(
        "--with-groq",
        action="store_true",
        help="Also compare grounded vs blind Groq keyword coverage",
    )
    args = parser.parse_args()

    fixture_path = Path(args.fixture)
    if not fixture_path.is_absolute():
        fixture_path = PROJECT_ROOT / fixture_path
    if not fixture_path.exists():
        raise SystemExit(f"Fixture not found: {fixture_path}")

    snippets = load_fixture(fixture_path)
    print(f"Loaded {len(snippets)} snippets from {fixture_path}\n")
    print("Deterministic analyzer coverage:")
    report = {
        "fixture": str(fixture_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deterministic": run_deterministic(snippets),
    }

    if args.with_groq:
        if not os.getenv("GROQ_API_KEY"):
            raise SystemExit("--with-groq requires GROQ_API_KEY")
        print("\nGrounded vs blind Groq:")
        report["groq"] = asyncio.run(run_with_groq(snippets))

    DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        DEFAULT_LOG_DIR
        / f"grounded_vs_blind_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {out_path}")

    hit_rate = report["deterministic"]["hit_rate"]
    if hit_rate < args.min_hit_rate:
        raise SystemExit(
            f"Hit-rate {hit_rate:.1%} below minimum {args.min_hit_rate:.0%}"
        )


if __name__ == "__main__":
    main()
