"""
scripts/eval_grounded_vs_blind.py
---------------------------------
Runs an evaluation across a small set of fixtures to compare:
1. "Blind" Groq (code only)
2. "Grounded" Groq (code + PolyMentor Static Analyzer Context)

This proves the productivity and accuracy gains of the ContextBuilder.
"""

import sys
import os
import asyncio
from pathlib import Path
import textwrap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import PolyMentorPipeline
from src.inference.context_builder import ContextBuilder

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "eval_fixtures"

async def evaluate_file(pipeline: PolyMentorPipeline, filepath: Path) -> str:
    print(f"Evaluating {filepath.name}...")
    
    with open(filepath, "r", encoding="utf-8") as f:
        code = f.read()

    ext = filepath.suffix.lower()
    language = "python" if ext == ".py" else "javascript"
    
    message = "Review this code and identify any bugs or bad practices."

    # 1. Blind Groq (No static analysis context)
    blind_response = await pipeline.chat(
        message=message,
        code=code,
        language=language,
        level="intermediate",
        analyzer_context=""
    )

    # 2. Grounded Groq (With ContextBuilder)
    analyzer_context = ContextBuilder.build_analyzer_context(code, language)
    grounded_response = await pipeline.chat(
        message=message,
        code=code,
        language=language,
        level="intermediate",
        analyzer_context=analyzer_context
    )

    # Build report section
    report = [
        f"## Fixture: `{filepath.name}`",
        f"\n### Code:",
        f"```{language}\n{code}\n```",
        f"\n### 🔍 Static Analyzer Raw Output",
        f"```text\n{analyzer_context or 'No issues found.'}\n```",
        f"\n### 🙈 Blind Groq Review (No Context)",
        f"**Bugs Found:** {len(blind_response.suspected_bugs)}",
        f"**Answer Extract:**",
        textwrap.indent(blind_response.answer, "> "),
        f"\n### 🎯 Grounded Groq Review (With Context)",
        f"**Bugs Found:** {len(grounded_response.suspected_bugs)}",
        f"**Answer Extract:**",
        textwrap.indent(grounded_response.answer, "> "),
        "\n---\n"
    ]
    return "\n".join(report)

async def main():
    if not FIXTURES_DIR.exists():
        print(f"Error: Fixtures directory not found -> {FIXTURES_DIR}")
        sys.exit(1)

    pipeline = PolyMentorPipeline()
    if not pipeline.api_key:
        print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    print("Starting Grounded vs Blind Evaluation...")
    report_content = [
        "# Grounded vs Blind Groq Evaluation",
        "This report compares the AI tutor's performance when relying solely on its intrinsic knowledge (Blind) versus when it is augmented with our `AdvancedCodeAnalyzer` (Grounded).\n",
        "---"
    ]

    for filepath in FIXTURES_DIR.glob("*.*"):
        if filepath.suffix in [".py", ".js"]:
            section = await evaluate_file(pipeline, filepath)
            report_content.append(section)

    report_path = Path("eval_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_content))

    print(f"\nEvaluation complete! Report saved to {report_path.absolute()}")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
