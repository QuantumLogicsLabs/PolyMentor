#!/usr/bin/env python3
"""
triage_pytest_failure.py
------------------------
Analyzes a pytest failure log with PolyMentor's Groq pipeline and writes a
sticky PR comment markdown file (pytest_triage_comment.md).

Always exits 0 so triage issues never mask the original red pytest job.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference.pipeline import PolyMentorPipeline
from src.inference.repo_parser import RepoParser

OUT_PATH = Path("pytest_triage_comment.md")
MAX_CHARS = 30000


@dataclass
class FailedTest:
    name: str
    file_path: str
    line_number: Optional[int] = None
    exception_type: str = "UnknownError"
    message: str = "Test failure detected"
    traceback_snippet: str = ""


@dataclass
class PytestReport:
    total_failures: int = 0
    total_errors: int = 0
    failed_tests: list[FailedTest] = field(default_factory=list)
    short_summary: str = ""
    raw_log_length: int = 0


def parse_pytest_log(log_text: str) -> PytestReport:
    """
    Parses unstructured pytest console output into structured diagnostic metrics,
    extracting failing test names, file locations, line numbers, and exception tracebacks.
    """
    report = PytestReport(raw_log_length=len(log_text))
    
    # Extract short test summary info if present
    summary_match = re.search(r"=+\s*short test summary info\s*=+\n(.*?)(?=\n=+|\Z)", log_text, re.DOTALL)
    if summary_match:
        report.short_summary = summary_match.group(1).strip()
        
    # Find failure descriptions from short summary or tracebacks
    # e.g.: ERROR tests/test_context_aware_chat.py - NameError: name 'ContextBuilder' is not defined
    # e.g.: FAILED tests/test_hybrid_mentor.py::test_format - AssertionError
    summary_lines = report.short_summary.splitlines() if report.short_summary else log_text.splitlines()
    for line in summary_lines:
        line = line.strip()
        if line.startswith("ERROR ") or line.startswith("FAILED "):
            is_error = line.startswith("ERROR ")
            if is_error:
                report.total_errors += 1
            else:
                report.total_failures += 1
                
            parts = line.split(" - ", 1)
            target_info = parts[0].replace("ERROR ", "").replace("FAILED ", "").strip()
            err_msg = parts[1].strip() if len(parts) > 1 else "Unknown test failure"
            
            # Extract exception type from err_msg if available
            exc_type = "Failure"
            if ":" in err_msg and not err_msg.startswith("http"):
                exc_type = err_msg.split(":", 1)[0].strip()
            elif err_msg.split()[0].endswith("Error") or err_msg.split()[0].endswith("Exception"):
                exc_type = err_msg.split()[0]
                
            # Parse file and line or test name
            file_path = target_info
            test_name = target_info
            line_num = None
            if "::" in target_info:
                file_path, test_name = target_info.split("::", 1)
            elif ":" in target_info:
                file_path, ln = target_info.split(":", 1)
                if ln.isdigit():
                    line_num = int(ln)
                    
            report.failed_tests.append(FailedTest(
                name=test_name,
                file_path=file_path,
                line_number=line_num,
                exception_type=exc_type,
                message=err_msg
            ))
            
    # If no failures found in summary, do fallback heuristic scan on tracebacks
    if not report.failed_tests and ("FAILURES" in log_text or "ERRORS" in log_text):
        report.total_failures = log_text.count("FAILED") + log_text.count("ERROR")
        
    return report


def extract_critical_traceback_tail(log_text: str, max_chars: int = 25000) -> tuple[str, bool]:
    """
    Intelligently extracts the most relevant diagnostic sections of a pytest execution log.
    When logs exceed token budgets, standard top-truncation clips off tracebacks and test summaries
    that appear at the end of output. This function prioritizes retaining the tail summary and error stack traces.

    Returns:
        tuple: (processed_log_text, was_truncated)
    """
    if len(log_text) <= max_chars:
        return log_text, False

    lines = log_text.splitlines(keepends=True)
    
    # Always try to keep the first 50 lines (test command setup, pytest environment, plugins)
    header = "".join(lines[:50])
    
    # Calculate budget remaining for the tail
    tail_budget = max_chars - len(header) - 100  # reserve space for truncation notice
    if tail_budget < 1000:
        return log_text[-max_chars:], True  # fallback to pure tail slice if header is massive

    # Collect lines from the bottom up until tail_budget is reached
    tail_lines = []
    current_len = 0
    for line in reversed(lines[50:]):
        if current_len + len(line) <= tail_budget:
            tail_lines.append(line)
            current_len += len(line)
        else:
            break
            
    tail = "".join(reversed(tail_lines))
    
    notice = "\n... [Mid-section of pytest log omitted to preserve critical test failure tracebacks within token budget] ...\n\n"
    return f"{header}{notice}{tail}", True


def ground_failure_with_source_code(report: PytestReport, repo_root: Optional[str] = None, max_files: int = 4) -> str:
    """
    Correlates failing test locations from the pytest diagnostic report with real repository source code.
    Reads test snippets and extracts structural AST symbols from dependent application modules
    to provide deep root-cause grounding for the AI triage engine.
    """
    if not report.failed_tests:
        return "Source code grounding: No specific failing test file paths identified in summary."

    if not repo_root:
        cwd = Path.cwd()
        if (cwd / "tests").exists() or (cwd / "src").exists() or (cwd / "pyproject.toml").exists():
            repo_root = str(cwd)
            
    if not repo_root or not Path(repo_root).exists():
        return "Source code grounding: Repository workspace root not found."

    context_parts = [f"Repository Source Code Grounding (Workspace: {Path(repo_root).name}):"]
    seen_files = set()
    files_processed = 0

    try:
        parser = RepoParser()
        for test_item in report.failed_tests:
            if files_processed >= max_files:
                context_parts.append(f"\n... [Remaining {len(report.failed_tests) - max_files} failed test files omitted from deep source inspection for context economy]")
                break

            file_p = Path(repo_root) / test_item.file_path
            if str(file_p) in seen_files:
                continue
            seen_files.add(str(file_p))
            
            if file_p.exists() and file_p.is_file():
                files_processed += 1
                context_parts.append(f"\n--- Grounded Source: {test_item.file_path} (Exception: {test_item.exception_type}) ---")
                
                try:
                    content_lines = file_p.read_text(encoding="utf-8", errors="replace").splitlines()
                    # If line number is known, extract around the failure point (±25 lines)
                    if test_item.line_number and 1 <= test_item.line_number <= len(content_lines):
                        start_idx = max(0, test_item.line_number - 25)
                        end_idx = min(len(content_lines), test_item.line_number + 15)
                        snippet_lines = content_lines[start_idx:end_idx]
                        context_parts.append(f"Snippet around line {test_item.line_number}:")
                        for idx, l_text in enumerate(snippet_lines, start=start_idx + 1):
                            prefix = "--> " if idx == test_item.line_number else "    "
                            context_parts.append(f"{prefix}{idx}: {l_text}")
                    else:
                        # Otherwise include first 60 lines of test file
                        context_parts.append("File preview (first 60 lines):")
                        context_parts.extend(content_lines[:60])
                        
                    # Extract AST dependencies using RepoParser
                    summary = parser.extract_repo_context(str(file_p), repo_root)
                    if summary and summary.get("related_files"):
                        rel_list = [str(rf) for rf in summary["related_files"][:3]]
                        context_parts.append("Dependent workspace modules: " + ", ".join(rel_list))
                        
                except Exception as read_err:
                    context_parts.append(f"Could not read source file snippet: {read_err}")

        return "\n".join(context_parts)
    except Exception as e:
        return f"Source code grounding skipped due to exception: {str(e)}"


def write_comment(body: str, out_path: Optional[Path] = None) -> None:
    target = out_path or OUT_PATH
    target.write_text(body.strip() + "\n", encoding="utf-8")
    print(f"Wrote triage comment to {target.absolute()}")




async def main() -> None:
    if len(sys.argv) < 2:
        write_comment(
            "## PolyMentor pytest triage\n\n"
            "Triage unavailable: no pytest log path was provided."
        )
        return

    log_path = Path(sys.argv[1])
    if not log_path.exists():
        write_comment(
            "## PolyMentor pytest triage\n\n"
            f"Triage unavailable: log file not found (`{log_path}`)."
        )
        return

    log_content = log_path.read_text(encoding="utf-8", errors="replace")
    if not log_content.strip():
        write_comment(
            "## PolyMentor pytest triage\n\n"
            "Triage unavailable: pytest log was empty."
        )
        return

    if len(log_content) > MAX_CHARS:
        log_content = (
            log_content[:MAX_CHARS] + "\n... [Log truncated due to size limit]"
        )

    if not os.getenv("GROQ_API_KEY"):
        write_comment(
            "## PolyMentor pytest triage\n\n"
            "Triage unavailable: `GROQ_API_KEY` is not set in repository secrets."
        )
        return

    print(f"Triaging pytest failure log ({len(log_content)} chars)...")

    try:
        pipeline = PolyMentorPipeline.from_groq()
        question = (
            "You are an expert CI failure triage engineer. Analyze this pytest "
            "failure log from the PolyMentor repository. Identify the most likely "
            "root cause, the files or tests involved, whether this looks like a "
            "real regression vs a flaky/environment issue, and give a short list "
            "of minimal fix steps. Be concrete and concise. Use markdown headings "
            "and bullet lists."
        )
        result = await pipeline.analyze(
            code=log_content,
            language="python",
            level="advanced",
            question=question,
        )
    except Exception as exc:  # noqa: BLE001 — never fail the CI job from triage
        write_comment(
            "## PolyMentor pytest triage\n\n"
            f"Triage unavailable: {type(exc).__name__}: {exc}"
        )
        return

    if result.status != "ok":
        write_comment(
            "## PolyMentor pytest triage\n\n"
            f"Triage unavailable: pipeline status `{result.status}`.\n\n"
            f"{result.answer or ''}"
        )
        return

    lines = [
        "## PolyMentor pytest fail triage",
        "",
        result.answer or "_No triage summary returned._",
    ]

    if result.suspected_bugs:
        lines.append("\n### Suspected issues")
        for bug in result.suspected_bugs:
            lines.append(f"- {bug}")

    if result.next_steps:
        lines.append("\n### Suggested fix steps")
        for step in result.next_steps:
            lines.append(f"- {step}")

    if result.lesson:
        lines.append(f"\n### Key takeaway\n{result.lesson}")

    lines.append(
        f"\n---\n*Triaged with {result.model} in {result.elapsed_ms:.0f}ms*"
    )
    write_comment("\n".join(lines))


if __name__ == "__main__":
    asyncio.run(main())
