#!/usr/bin/env python3
"""
analyze_pr.py
-------------
Analyzes a Pull Request diff using PolyMentor's Groq pipeline and generates a markdown comment.
"""

import asyncio
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference.pipeline import PolyMentorPipeline
from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer
from src.inference.repo_parser import RepoParser
from scripts.analyze_file import infer_language_from_filename, find_repo_root




@dataclass
class DiffHunk:
    """Represents a parsed file diff hunk with extracted code modifications."""
    file_path: str
    language: str
    added_code: str
    raw_diff: str


def parse_pr_diff(diff_content: str) -> list[DiffHunk]:
    """
    Parses a standard git unified diff into structured per-file DiffHunks.
    Extracts added lines to enable deterministic static analysis on new code.
    """
    if not diff_content or not diff_content.strip():
        return []

    hunks = []
    # Split diff by file headers (diff --git or +++ / --- blocks)
    file_sections = re.split(r'(?:^|\n)diff --git a/.*? b/|(?:\n\+\+\+ (?:b/)?([^\n]+))', diff_content)
    
    current_file = None
    current_diff_lines = []
    added_lines = []

    for line in diff_content.splitlines():
        if line.startswith("diff --git a/"):
            if current_file and (current_diff_lines or added_lines):
                lang = infer_language_from_filename(current_file)
                hunks.append(DiffHunk(
                    file_path=current_file,
                    language=lang,
                    added_code="\n".join(added_lines),
                    raw_diff="\n".join(current_diff_lines)
                ))
                current_diff_lines = []
                added_lines = []
            parts = line.split(" b/")
            if len(parts) == 2:
                current_file = parts[1].strip()
            else:
                current_file = line.split()[-1].replace("a/", "").replace("b/", "").strip()
            current_diff_lines.append(line)
        elif line.startswith("+++ b/") or line.startswith("+++ "):
            if not current_file or current_file == "dev/null":
                current_file = line.replace("+++ b/", "").replace("+++ ", "").strip()
            current_diff_lines.append(line)
        else:
            if current_file:
                current_diff_lines.append(line)
                if line.startswith("+") and not line.startswith("+++"):
                    added_lines.append(line[1:])

    if current_file and (current_diff_lines or added_lines):
        lang = infer_language_from_filename(current_file)
        hunks.append(DiffHunk(
            file_path=current_file,
            language=lang,
            added_code="\n".join(added_lines),
            raw_diff="\n".join(current_diff_lines)
        ))

    return hunks


@dataclass
class HunkAnalysisResult:
    """Combines a parsed diff hunk with its deterministic static analysis findings."""
    hunk: DiffHunk
    static_summary: dict
    has_bugs: bool
    quality_score: int


def analyze_hunks_static(hunks: list[DiffHunk]) -> list[HunkAnalysisResult]:
    """
    Runs deterministic static analysis on newly added code in PR diff hunks.
    Provides instant verification and grounding before calling LLM inference.
    """
    analyzer = AdvancedCodeAnalyzer()
    results = []
    
    for hunk in hunks:
        if not hunk.added_code.strip() or hunk.language not in {"python", "javascript", "typescript", "java", "cpp"}:
            results.append(HunkAnalysisResult(
                hunk=hunk,
                static_summary={"supported": False, "total_errors": 0, "quality_score": 100},
                has_bugs=False,
                quality_score=100
            ))
            continue
            
        summary = analyzer.analyze(hunk.added_code, hunk.language)
        errors_count = summary.get("total_errors", 0)
        score = summary.get("quality_score", 100)
        
        results.append(HunkAnalysisResult(
            hunk=hunk,
            static_summary=summary,
            has_bugs=errors_count > 0,
            quality_score=score
        ))
        
    return results


def extract_pr_repo_context(hunk_results: list[HunkAnalysisResult], repo_root: Optional[str] = None) -> str:
    """
    Synthesizes structural AST repository symbols and dependency signatures for modified files.
    Enables LLM inference to review architecture impact across inter-file dependencies.
    """
    if not repo_root:
        cwd = Path.cwd()
        if (cwd / ".git").exists() or (cwd / "pyproject.toml").exists() or (cwd / "package.json").exists():
            repo_root = str(cwd)
        elif hunk_results:
            first_file = Path(hunk_results[0].hunk.file_path)
            if first_file.exists():
                repo_root = find_repo_root(first_file)
                
    if not repo_root or not Path(repo_root).exists():
        return "Repository context: Isolated PR diff review (no local repository workspace found)."

    try:
        parser = RepoParser()
        context_parts = [f"Repository Architecture Context (Workspace: {Path(repo_root).name}):"]
        
        for hr in hunk_results[:10]:  # Limit to top 10 modified files for context budget
            file_p = Path(repo_root) / hr.hunk.file_path
            if file_p.exists() and file_p.is_file() and hr.hunk.language in {"python", "javascript", "typescript", "java", "cpp"}:
                summary = parser.extract_repo_context(str(file_p), repo_root)
                if summary:
                    context_parts.append(f"\n--- Structural context for {hr.hunk.file_path} ---")
                    if summary.get("symbols"):
                        context_parts.append("Symbols defined in repository: " + ", ".join(s["name"] for s in summary["symbols"][:8]))
                    if summary.get("related_files"):
                        context_parts.append("Dependent / related workspace files: " + ", ".join(str(rf) for rf in summary["related_files"][:5]))
                        
        return "\n".join(context_parts)
    except Exception as e:
        return f"Repository context abstraction skipped: {str(e)}"


def truncate_diff_budget(hunk_results: list[HunkAnalysisResult], max_chars: int = 25000) -> tuple[str, bool, list[str]]:
    """
    Intelligently packs PR diff hunks into a fixed character token budget.
    Prioritizes files containing deterministic static analysis bugs, followed by core source code,
    while deferring or omitting bulky lockfiles and config dumps.

    Returns:
        tuple: (packed_diff_string, was_truncated, list_of_dropped_files)
    """
    if not hunk_results:
        return "", False, []

    # Sort hunks by priority: (has_bugs -> core code file -> lower quality score -> shorter length)
    def priority_key(hr: HunkAnalysisResult):
        is_core = hr.hunk.language in {"python", "javascript", "typescript", "java", "cpp", "go", "rust"}
        is_lock = any(hr.hunk.file_path.endswith(l) for l in [".lock", "package-lock.json", "poetry.lock", "yarn.lock"])
        return (0 if is_lock else 1, 1 if hr.has_bugs else 0, 1 if is_core else 0, -hr.quality_score)

    sorted_hunks = sorted(hunk_results, key=priority_key, reverse=True)
    
    packed_lines = []
    current_chars = 0
    was_truncated = False
    dropped_files = []

    for hr in sorted_hunks:
        hunk_len = len(hr.hunk.raw_diff)
        if current_chars + hunk_len <= max_chars:
            packed_lines.append(f"--- Modifying: {hr.hunk.file_path} (Quality Score: {hr.quality_score}/100) ---")
            packed_lines.append(hr.hunk.raw_diff)
            current_chars += hunk_len + 80
        else:
            was_truncated = True
            dropped_files.append(hr.hunk.file_path)
            # Try appending a truncated preview if there's still meaningful room
            remaining = max_chars - current_chars
            if remaining > 500:
                preview_lines = hr.hunk.raw_diff[:remaining - 150]
                packed_lines.append(f"--- Modifying: {hr.hunk.file_path} (Truncated due to token budget) ---")
                packed_lines.append(preview_lines + "\n... [Remaining hunk truncated for token budget preservation]")
                current_chars = max_chars

    return "\n\n".join(packed_lines), was_truncated, dropped_files


async def main():
    if len(sys.argv) < 2:



        print("Usage: python analyze_pr.py <path_to_diff>")
        sys.exit(1)


    diff_path = Path(sys.argv[1])
    if not diff_path.exists():
        print(f"Diff file not found: {diff_path}")
        sys.exit(1)

    diff_content = diff_path.read_text(encoding="utf-8")
    if not diff_content.strip():
        Path("pr_comment.md").write_text("No changes found in the diff.")
        sys.exit(0)
        
    # Optional: Truncate diff if it's too massive for the model context
    # Groq's llama-3.3-70b-versatile has a very large context, but just in case
    max_chars = 30000 
    if len(diff_content) > max_chars:
        diff_content = diff_content[:max_chars] + "\n... [Diff truncated due to size limit]"

    print(f"Analyzing PR diff ({len(diff_content)} chars)...")
    
    pipeline = PolyMentorPipeline.from_groq()
    
    question = (
        "You are an expert code reviewer. Please review this pull request diff. "
        "Identify any bugs, security vulnerabilities, or performance issues. "
        "Suggest improvements and provide a brief summary of the changes."
    )
    
    result = await pipeline.analyze(
        code=diff_content,
        language="diff",
        level="advanced",
        question=question
    )

    if result.status != "ok":
        Path("pr_comment.md").write_text(f"PolyMentor encountered an error: {result.status}\n{result.answer}")
        sys.exit(0)

    comment_lines = []
    comment_lines.append("## 🤖 PolyMentor PR Review\n")
    comment_lines.append(result.answer)
    
    if result.suspected_bugs:
        comment_lines.append("\n### 🐛 Suspected Bugs")
        for bug in result.suspected_bugs:
            comment_lines.append(f"- {bug}")
            
    if result.lesson:
        comment_lines.append(f"\n### 💡 Key Takeaway\n{result.lesson}")
        
    if result.next_steps:
        comment_lines.append("\n### 🎯 Suggested Action Items")
        for step in result.next_steps:
            comment_lines.append(f"- {step}")
            
    if result.fixed_code:
        comment_lines.append(f"\n### ✨ Suggested Fix\n```diff\n{result.fixed_code}\n```")
        
    comment_lines.append(f"\n---\n*Analyzed with {result.model} in {result.elapsed_ms:.0f}ms*")
    
    out_path = Path("pr_comment.md")
    out_path.write_text("\n".join(comment_lines), encoding="utf-8")
    print(f"Successfully wrote PR comment to {out_path.absolute()}")

if __name__ == "__main__":
    asyncio.run(main())
