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
import sys
import os
from pathlib import Path

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


async def analyze_file(file_path: str, language: str, level: str, model: str):
    path = Path(file_path)
    if not path.is_file():
        print(f"Error: File '{file_path}' does not exist.")
        sys.exit(1)

    if language.lower() == "auto" or not language:
        language = infer_language_from_filename(str(path))


    try:
        code_content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        sys.exit(1)

    print(f"Analyzing {file_path} ({len(code_content)} chars) with {model}...\n")
    
    pipeline = PolyMentorPipeline.from_groq(model=model)
    
    try:
        result = await pipeline.analyze(
            code=code_content,
            language=language,
            level=level,
            question="Review this code, identify likely bugs, teach the concept, and suggest improvements."
        )
        
        if result.status != "ok":
            print(f"Error from PolyMentorPipeline: {result.status}")
            print(result.answer)
            return

        print("=" * 60)
        print("Code Review & Analysis")
        print("=" * 60)
        
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

    except Exception as exc:
        print(f"Analysis failed: {exc}")
        logger.error("Analysis failed: %s", exc, exc_info=True)


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
    args = parser.parse_args()

    asyncio.run(analyze_file(args.file, args.language, args.level, args.model))


if __name__ == "__main__":
    main()
