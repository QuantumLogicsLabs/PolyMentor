"""
scripts/pr_reviewer.py
----------------------
Automatically reviews Pull Request changes using PolyMentor pipeline.
Designed to be run from within a GitHub Action.
"""

import sys
import os
import asyncio
import subprocess
from pathlib import Path

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

def get_changed_files(base_sha: str, head_sha: str) -> list[str]:
    """Gets a list of changed files from git diff."""
    cmd = ["git", "diff", "--name-only", f"{base_sha}..{head_sha}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Git diff error: {result.stderr}")
        return []
    
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]

def get_file_diff(base_sha: str, head_sha: str, filepath: str) -> str:
    """Gets the unified diff for a specific file."""
    cmd = ["git", "diff", f"{base_sha}..{head_sha}", "--", filepath]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout

async def main():
    base_sha = os.environ.get("BASE_SHA")
    head_sha = os.environ.get("HEAD_SHA")
    pr_number = os.environ.get("PR_NUMBER")

    if not base_sha or not head_sha:
        print("Error: BASE_SHA and HEAD_SHA environment variables must be set.", file=sys.stderr)
        sys.exit(1)

    pipeline = PolyMentorPipeline()
    if not pipeline.api_key:
        print("Error: GROQ_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    changed_files = get_changed_files(base_sha, head_sha)
    
    # Filter for supported files
    supported_files = [f for f in changed_files if Path(f).suffix.lower() in EXT_MAP]
    
    if not supported_files:
        print(f"## 🤖 PolyMentor PR Review\n\nNo supported source code files were modified in this PR.")
        return

    print(f"## 🤖 PolyMentor PR Review\n")
    print(f"*I've analyzed the {len(supported_files)} supported file(s) changed in this PR.* 🚀\n")

    for filepath in supported_files:
        ext = Path(filepath).suffix.lower()
        language = EXT_MAP[ext]
        
        # Read current file contents if it still exists (might be deleted)
        if not os.path.exists(filepath):
            continue
            
        with open(filepath, "r", encoding="utf-8") as f:
            code = f.read()

        file_diff = get_file_diff(base_sha, head_sha, filepath)
        if not file_diff.strip():
            continue

        # Run Static Analysis
        analyzer_context = ContextBuilder.build_analyzer_context(code, language)
        
        # Build prompt message focusing on the diff
        message = (
            f"Review the following Git diff for `{filepath}`. Focus on identifying "
            f"bugs, anti-patterns, or logic errors specifically in the added/modified lines. "
            f"If the code looks good, just say so. Keep it concise.\n\n"
            f"Git Diff:\n```diff\n{file_diff}\n```"
        )

        response = await pipeline.chat(
            message=message,
            code=code,
            language=language,
            level="intermediate",  # Default to intermediate for PR reviews
            analyzer_context=analyzer_context
        )

        print(f"### 📄 `{filepath}`")
        if response.status == "ok":
            if response.suspected_bugs:
                print("#### ⚠️ Suspected Bugs in Changes:")
                for bug in response.suspected_bugs:
                    print(f"- {bug}")
            
            print(f"\n#### 📝 AI Analysis:")
            print(response.answer)
            
            if response.fixed_code:
                print(f"\n#### ✨ Suggested Fix:")
                print(f"```{language}\n{response.fixed_code}\n```")
        else:
            print(f"> Error generating review for this file: {response.status}")
            
        print("\n---\n")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
