#!/usr/bin/env python3
"""
analyze_pr.py
-------------
Analyzes a Pull Request diff using PolyMentor's Groq pipeline and generates a markdown comment.
"""

import argparse
import asyncio
import json
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
    
    Filters out minor low-severity style nits when determining `has_bugs`, only asserting
    actionable bug status when critical, high, or medium severity issues are present.
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
        actionable_bugs = (
            summary.get("critical_count", 0)
            + summary.get("high_count", 0)
            + summary.get("medium_count", 0)
        )
        
        results.append(HunkAnalysisResult(
            hunk=hunk,
            static_summary=summary,
            has_bugs=actionable_bugs > 0,
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


def generate_pr_comment_md(response, hunk_results: list[HunkAnalysisResult], dropped_files: list[str]) -> str:
    """
    Synthesizes an enterprise-grade GitHub PR markdown review comment combining
    deterministic static analysis findings, quality gate scorecards, and LLM guidance.
    """
    lines = ["# 🤖 PolyMentor PR Review & Quality Gate\n"]
    
    # Calculate aggregate scores and total actionable bugs
    total_errors = sum(hr.static_summary.get("total_errors", 0) for hr in hunk_results)
    total_bugs = sum(
        (hr.static_summary.get("critical_count", 0)
         + hr.static_summary.get("high_count", 0)
         + hr.static_summary.get("medium_count", 0))
        if "critical_count" in hr.static_summary else hr.static_summary.get("total_errors", 0)
        for hr in hunk_results
    )
    supported_hunks = [hr for hr in hunk_results if hr.static_summary.get("supported", False)]
    avg_score = int(sum(hr.quality_score for hr in supported_hunks) / len(supported_hunks)) if supported_hunks else 100
    
    # Risk assessment badge
    if total_errors > 5 or avg_score < 60:
        risk_badge = "🔴 **Critical Risk** — Immediate fixes required before merge."
    elif total_errors > 0 or avg_score < 80:
        risk_badge = "🟡 **Medium Risk** — Review suspected bugs and quality recommendations."
    else:
        risk_badge = "🟢 **Low Risk / Clean** — Code meets quality threshold standards."
        
    lines.append(f"### 🛡️ Risk & Quality Assessment\n{risk_badge}\n")
    lines.append(f"- **Aggregate Quality Score:** `{avg_score}/100`")
    lines.append(f"- **Deterministic Bug Findings:** `{total_errors} issue(s)` across `{len(hunk_results)} file(s)` analyzed.\n")
    
    # Static Analysis Table
    if supported_hunks:
        lines.append("### 📊 File Quality Scorecard\n")
        lines.append("| File Path | Language | Quality Score | Static Errors | Status |")
        lines.append("| :--- | :---: | :---: | :---: | :---: |")
        for hr in hunk_results:
            err_count = hr.static_summary.get("total_errors", 0)
            status_icon = "❌ Failed" if err_count > 0 else "✅ Passed"
            lines.append(f"| `{hr.hunk.file_path}` | `{hr.hunk.language}` | `{hr.quality_score}/100` | `{err_count}` | {status_icon} |")
        lines.append("")

    if total_errors > 0:
        lines.append("### 🐛 Deterministic Static Analysis Findings\n")
        for hr in hunk_results:
            errors = hr.static_summary.get("errors", [])
            if errors:
                lines.append(f"**In `{hr.hunk.file_path}`:**")
                for err in errors[:5]:  # Cap at top 5 per file
                    rule = err.get("rule", "bug_pattern")
                    msg = err.get("message", "Issue detected")
                    lines.append(f"- `[{rule}]`: {msg}")
        lines.append("")

    lines.append("### 🧠 Senior AI Mentor Architecture & Logic Review\n")
    lines.append(response.answer)
    
    if response.suspected_bugs:
        lines.append("\n### ⚠️ Potential Logical Risks & Vulnerabilities")
        for bug in response.suspected_bugs:
            lines.append(f"- {bug}")
            
    if response.lesson:
        lines.append(f"\n### 💡 Mentor Pedagogical Insight\n{response.lesson}")
        
    if response.next_steps:
        lines.append("\n### 🎯 Recommended Action Items & Test Gaps")
        for step in response.next_steps:
            lines.append(f"- {step}")
            
    if response.fixed_code:
        lines.append(f"\n### ✨ Suggested Code Refactoring\n```diff\n{response.fixed_code}\n```")
        
    if dropped_files:
        lines.append(f"\n> ℹ️ *Note: {len(dropped_files)} file(s) omitted from deep LLM context to preserve token budget ({', '.join(dropped_files[:3])}...).*")
        
    lines.append(f"\n---\n*Grounded review completed using **{response.model}** in **{response.elapsed_ms:.0f}ms** (Static analysis grounding enabled)*")
    return "\n".join(lines)


async def run_pr_review(
    diff_file: str,
    output_md: str = "pr_comment.md",
    json_summary: Optional[str] = None,
    fail_on_bugs: bool = False,
    min_score: Optional[float] = None,
    pipeline: Optional[PolyMentorPipeline] = None
) -> tuple[int, dict]:
    """
    Executes the grounded PR review pipeline, returning an exit code and summary dictionary.
    Exit codes: 0=pass, 1=system error, 2=bugs found (if --fail-on-bugs), 3=score below min_score.
    """
    diff_path = Path(diff_file)
    if not diff_path.exists():
        print(f"Error: Diff file not found at {diff_path}")
        return 1, {"error": "Diff file not found"}

    diff_content = diff_path.read_text(encoding="utf-8")
    out_path = Path(output_md)

    if not diff_content.strip():
        out_path.write_text("No changes found in the diff.", encoding="utf-8")
        if json_summary:
            Path(json_summary).write_text(json.dumps({"status": "empty_diff", "quality_score": 100, "total_errors": 0}), encoding="utf-8")
        return 0, {"status": "empty_diff", "quality_score": 100, "total_errors": 0}

    print(f"Parsing PR diff ({len(diff_content)} chars)...")
    hunks = parse_pr_diff(diff_content)
    print(f"Extracted {len(hunks)} file modification hunk(s). Running deterministic static analysis...")
    
    hunk_results = analyze_hunks_static(hunks)
    
    # Calculate quality metrics
    total_errors = sum(hr.static_summary.get("total_errors", 0) for hr in hunk_results)
    supported_hunks = [hr for hr in hunk_results if hr.static_summary.get("supported", False)]
    avg_score = int(sum(hr.quality_score for hr in supported_hunks) / len(supported_hunks)) if supported_hunks else 100

    print(f"Static analysis complete: Aggregate Quality Score = {avg_score}/100, Total Static Errors = {total_errors}")
    
    # Extract AST repo grounding and pack token budget
    repo_context = extract_pr_repo_context(hunk_results)
    packed_diff, was_truncated, dropped_files = truncate_diff_budget(hunk_results)
    
    if not pipeline:
        pipeline = PolyMentorPipeline.from_groq()

    question = (
        "You are a senior expert software architect reviewing this pull request diff. "
        f"{repo_context}\n"
        "Please review the code changes below. Identify structural bugs, concurrency hazards, "
        "security vulnerabilities, or architectural degradation. Highlight test gaps and suggest improvements."
    )
    
    result = await pipeline.analyze(
        code=packed_diff,
        language="diff",
        level="advanced",
        question=question
    )

    if result.status != "ok":
        err_msg = f"PolyMentor inference error: {result.status}\n{result.answer}"
        out_path.write_text(err_msg, encoding="utf-8")
        return 1, {"status": "error", "error": err_msg}

    # Generate enterprise markdown report
    md_comment = generate_pr_comment_md(result, hunk_results, dropped_files)
    out_path.write_text(md_comment, encoding="utf-8")
    print(f"Successfully wrote PR markdown review to {out_path.absolute()}")

    summary_data = {
        "status": "ok",
        "files_analyzed": len(hunks),
        "total_static_errors": total_errors,
        "aggregate_quality_score": avg_score,
        "model": result.model,
        "elapsed_ms": result.elapsed_ms,
        "was_truncated": was_truncated,
        "dropped_files": dropped_files
    }
    
    if json_summary:
        Path(json_summary).write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        print(f"Wrote machine-readable PR review summary to {json_summary}")

    # Evaluate Quality Gates
    if fail_on_bugs and total_errors > 0:
        print(f"Quality Gate Failed: --fail-on-bugs triggered ({total_errors} deterministic static error(s) found).")
        return 2, summary_data
    if min_score is not None and avg_score < min_score:
        print(f"Quality Gate Failed: Aggregate Quality Score ({avg_score}) below threshold ({min_score}).")
        return 3, summary_data

    return 0, summary_data


async def main():
    parser = argparse.ArgumentParser(description="PolyMentor Automated PR Diff Reviewer & Quality Gate")
    parser.add_argument("diff_file", help="Path to the unified pull request diff text file")
    parser.add_argument("--output", "-o", default="pr_comment.md", help="Path to save the generated markdown comment")
    parser.add_argument("--json-summary", "-j", help="Path to save a machine-readable JSON evaluation summary")
    parser.add_argument("--fail-on-bugs", action="store_true", help="Exit with code 2 if deterministic static errors are found")
    parser.add_argument("--min-score", type=float, help="Exit with code 3 if aggregate quality score falls below threshold")
    
    args = parser.parse_args()
    exit_code, _ = await run_pr_review(
        diff_file=args.diff_file,
        output_md=args.output,
        json_summary=args.json_summary,
        fail_on_bugs=args.fail_on_bugs,
        min_score=args.min_score
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
