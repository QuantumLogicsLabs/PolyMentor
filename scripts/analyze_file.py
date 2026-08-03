#!/usr/bin/env python3
"""
analyze_file.py
---------------
One-shot CLI script to analyze a local file using PolyMentor's Groq pipeline.

Usage:
    python scripts/analyze_file.py <path/to/file> [--language python] [--level beginner]
"""

import argparse
import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Optional, Any


# Add project root to sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.pipeline import DEFAULT_MODEL, PolyMentorPipeline
from src.utils.logger import get_logger

logger = get_logger(__name__)

EXTENSION_LANGUAGE_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".h": "cpp",
    ".c": "cpp",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".html": "html",
    ".css": "css",
}


def infer_language_from_filename(filename: str, default_language: str = "python") -> str:
    """Infer programming language from file extension for accurate analysis grounding."""
    ext = Path(filename).suffix.lower()
    return EXTENSION_LANGUAGE_MAP.get(ext, default_language)


def find_repo_root(target_path: Path) -> str | None:
    """Traverse parent directories to find the repository root (.git or project config)."""
    curr = target_path.resolve()
    if curr.is_file():
        curr = curr.parent
    for parent in [curr] + list(curr.parents):
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists() or (parent / "package.json").exists():
            return str(parent)
    return None


async def analyze_file(file_path: str, language: str, level: str, model: str, json_output: bool = False, pipeline: Optional[PolyMentorPipeline] = None) -> Any:
    path = Path(file_path)
    if not path.is_file():
        if json_output:
            print(json.dumps({"error": f"File '{file_path}' does not exist."}))
        else:
            print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    if language.lower() == "auto" or not language:
        language = infer_language_from_filename(str(path))


    try:
        code_content = path.read_text(encoding="utf-8")
    except Exception as e:
        if json_output:
            print(json.dumps({"error": f"Error reading file '{file_path}': {e}"}))
        else:
            print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)

    repo_root = find_repo_root(path)
    rel_path = str(path.resolve().relative_to(Path(repo_root))) if repo_root else path.name

    repo_context = None
    try:
        from src.inference.repo_parser import RepoParser
        repo_parser = RepoParser()
        repo_context = repo_parser.extract_repo_context(
            code=code_content,
            language=language,
            root_dir=repo_root,
            file_path=rel_path,
        )
        if repo_root and not json_output:
            print(f"[Grounding] Connected repo context: {repo_root} (target: {rel_path})")
    except Exception as e:
        logger.debug("Failed to extract repository context: %s", e)

    if not json_output:
        print(f"Analyzing {file_path} ({len(code_content)} chars, language: {language}) with {model}...\n")
    
    if pipeline is None:
        pipeline = PolyMentorPipeline.from_groq(model=model)
    
    try:
        result = await pipeline.analyze(
            code=code_content,
            language=language,
            level=level,
            question="Review this code, identify likely bugs, teach the concept, and suggest improvements.",
            repo=repo_context,
        )

        
        if json_output:
            payload = {
                "file": str(path),
                "language": language,
                "status": result.status,
                "answer": result.answer,
                "suspected_bugs": result.suspected_bugs,
                "lesson": result.lesson,
                "next_steps": result.next_steps,
                "fixed_code": result.fixed_code,
                "model": result.model,
                "elapsed_ms": result.elapsed_ms,
                "grounded": getattr(result, "grounded", False),
                "static_analysis_summary": getattr(result, "static_analysis_summary", None),
            }
            print(json.dumps(payload, indent=2))
            return result

        if result.status != "ok":
            print(f"Error from PolyMentorPipeline: {result.status}")
            print(result.answer)
            return result

        print("=" * 60)
        print("Code Review & Analysis")
        print("=" * 60)
        
        if result.static_analysis_summary and result.static_analysis_summary.get("supported"):
            summary = result.static_analysis_summary
            score = summary.get("quality_score", 0)
            err_count = summary.get("total_errors", 0)
            print(f"\n[Static Analysis Grounding] Quality Score: {score}/100 | Deterministic Issues: {err_count}")
            for err in summary.get("errors", []):
                cat = err.get("category", "issue")
                sev = err.get("severity", "medium").upper()
                line = err.get("line", "?")
                msg = err.get("message", "")
                sug = err.get("suggestion", "")
                print(f"  * Line {line} [{sev}] ({cat}): {msg}")
                if sug:
                    print(f"    -> Verified Suggestion: {sug}")
            print("-" * 60)
        
        print(f"\n{result.answer}\n")

        
        if result.suspected_bugs:
            print("-" * 60)
            print("Suspected Bugs:")
            for bug in result.suspected_bugs:
                print(f"- {bug}")
                
        if result.lesson:
            print("-" * 60)
            print(f"Lesson:\n{result.lesson}")
            
        if result.next_steps:
            print("-" * 60)
            print("Next Steps:")
            for step in result.next_steps:
                print(f"- {step}")
                
        if result.fixed_code:
            print("-" * 60)
            print("Suggested Fix:\n")
            print(result.fixed_code)

        print("-" * 60)
        print(f"Model: {result.model} | Time: {result.elapsed_ms:.0f} ms")
        return result

    except Exception as exc:
        if json_output:
            print(json.dumps({"status": "error", "error": str(exc)}))
        else:
            print(f"Analysis failed: {exc}")
        logger.error("Analysis failed: %s", exc, exc_info=True)
        return None


def check_quality_gates(result: Any, fail_on_bugs: bool = False, min_score: float = 0.0, json_output: bool = False) -> int:
    """Evaluate analysis results against quality gate thresholds and return appropriate exit code."""
    if result is None or getattr(result, "status", "error") != "ok":
        return 1

    if fail_on_bugs:
        bugs = getattr(result, "suspected_bugs", []) or []
        summary = getattr(result, "static_analysis_summary", None) or {}
        static_errors = summary.get("total_errors", 0) if summary.get("supported") else 0
        if bugs or static_errors > 0:
            if not json_output:
                print(f"\n[Quality Gate Failure] Found {len(bugs)} suspected bug(s) and {static_errors} static error(s).")
            return 2

    if min_score > 0:
        summary = getattr(result, "static_analysis_summary", None) or {}
        if summary.get("supported"):
            score = summary.get("quality_score", 0)
            if score < min_score:
                if not json_output:
                    print(f"\n[Quality Gate Failure] Quality score {score}/100 is below minimum threshold of {min_score}/100.")
                return 3
        else:
            if not json_output:
                print(f"\n[Quality Gate Warning] Static analysis not supported for language; skipping minimum score evaluation.")

    if not json_output and (fail_on_bugs or min_score > 0):
        print("\n[Quality Gate] Passed all configured quality thresholds.")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Analyze a local file using Groq-powered PolyMentor.")
    parser.add_argument("file", help="Path to the file to analyze.")
    parser.add_argument("--language", default="auto", help="Programming language or 'auto' to infer from extension.")
    parser.add_argument(
        "--level",
        default="intermediate",
        choices=["beginner", "intermediate", "advanced"],
        help="Explanation depth.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Groq model name.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Output machine-readable JSON.")
    parser.add_argument("--fail-on-bugs", action="store_true", help="Exit with code 2 if any suspected bugs or static errors are found.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Exit with code 3 if quality score falls below threshold.")
    args = parser.parse_args()

    result = asyncio.run(analyze_file(args.file, args.language, args.level, args.model, json_output=args.json_output))
    exit_code = check_quality_gates(result, fail_on_bugs=args.fail_on_bugs, min_score=args.min_score, json_output=args.json_output)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
