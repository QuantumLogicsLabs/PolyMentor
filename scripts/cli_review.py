"""
scripts/cli_review.py
---------------------
One-shot CLI to analyze a file locally and get a Groq-powered review 
using the PolyMentor pipeline and AdvancedCodeAnalyzer, without needing 
to start the frontend or API server.

Usage:
    python scripts/cli_review.py path/to/your/file.py --level intermediate
"""

import sys
import os
import argparse
import asyncio
import json
from pathlib import Path

# Add project root to python path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.inference.pipeline import PolyMentorPipeline
from src.inference.context_builder import ContextBuilder

# Map extensions to PolyMentor language names
EXT_MAP = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".cs": "csharp",
    ".go": "go",
    ".rs": "rust",
    ".php": "php",
    ".rb": "ruby",
    ".swift": "swift",
    ".kt": "kotlin",
    ".sql": "sql",
    ".html": "html",
    ".css": "css",
}

async def main():
    parser = argparse.ArgumentParser(description="PolyMentor CLI Code Reviewer")
    parser.add_argument("file", help="Path to the source code file to review")
    parser.add_argument("--level", choices=["beginner", "intermediate", "advanced"], default="intermediate", help="Target learner level")
    parser.add_argument("--message", default="Please review this code, point out any bugs, and suggest improvements.", help="Custom instructions for the AI")
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists() or not file_path.is_file():
        print(f"Error: File not found -> {file_path}")
        sys.exit(1)

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    ext = file_path.suffix.lower()
    language = EXT_MAP.get(ext, "python")  # Default to python if unknown

    print(f"Analyzing {file_path.name} as {language} (Level: {args.level})...")
    
    # 1. Run Static Analysis
    analyzer_context = ContextBuilder.build_analyzer_context(code, language)
    
    if analyzer_context:
        print("\n--- Static Analysis Found ---")
        print(analyzer_context)
        print("-----------------------------\n")
    else:
        print("No static issues found, waiting for AI review...\n")

    # 2. Get AI Review
    pipeline = PolyMentorPipeline()
    
    if not pipeline.api_key:
        print("Error: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)

    print("Generating AI Review...\n")
    
    response = await pipeline.chat(
        message=args.message,
        code=code,
        language=language,
        level=args.level,
        analyzer_context=analyzer_context
    )

    if response.status != "ok":
        print(f"Pipeline Error: {response.status}")
        print(response.answer)
        sys.exit(1)

    # 3. Print Results nicely
    print("=" * 60)
    print("🤖 AI REVIEW")
    print("=" * 60)
    
    if response.suspected_bugs:
        print("\n🐛 SUSPECTED BUGS:")
        for bug in response.suspected_bugs:
            print(f"  - {bug}")
            
    print("\n📝 EXPLANATION:")
    print(response.answer)
    
    if response.fixed_code:
        print("\n✨ SUGGESTED FIX:")
        print(f"```{language}")
        print(response.fixed_code)
        print("```")
        
    if response.lesson:
        print("\n🎓 KEY LESSON:")
        print(response.lesson)
        
    if response.next_steps:
        print("\n🚀 NEXT STEPS:")
        for step in response.next_steps:
            print(f"  - {step}")

    print(f"\n[Completed in {response.elapsed_ms/1000:.2f}s]")

if __name__ == "__main__":
    # Windows asyncio bug workaround
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
