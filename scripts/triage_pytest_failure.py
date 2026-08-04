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


def generate_triage_comment_md(
    report: PytestReport,
    ai_summary: Optional[str] = None,
    suspected_bugs: Optional[list[str]] = None,
    next_steps: Optional[list[str]] = None,
    lesson: Optional[str] = None,
    model_name: str = "local-diagnostic",
    elapsed_ms: float = 0.0,
    error_msg: Optional[str] = None
) -> str:
    """
    Synthesizes an enterprise-grade Markdown diagnosis report for sticky PR commenting,
    incorporating visual status badges, failure scorecards, and actionable AI remediations.
    """
    # Determine severity & badge
    total_issues = report.total_failures + report.total_errors
    if report.total_errors > 0:
        status_badge = "🟡 **Collection / Import Error** (Test Setup Failure)"
    elif total_issues > 0:
        status_badge = "🔴 **Test Suite Regression** (Code Assertion Failure)"
    else:
        status_badge = "🟢 **No Failures Detected** (Clean CI Run)"

    lines = [
        "## 🧪 PolyMentor Automated Pytest Triage",
        "",
        f"**CI Diagnostic Assessment:** {status_badge}",
        "",
        "### 📊 Test Suite Failure Scorecard",
        "| Metric | Count | Diagnostic Assessment |",
        "| :--- | :---: | :--- |",
        f"| **Runtime Test Failures** | **{report.total_failures}** | {'🚨 Immediate logic regression check required' if report.total_failures > 0 else '✅ No failed assertions'} |",
        f"| **Collection / Setup Errors** | **{report.total_errors}** | {'⚠️ Module import or syntax issue detected' if report.total_errors > 0 else '✅ Healthy test collection'} |",
        f"| **Analyzed Log Size** | **{report.raw_log_length}** chars | {'📉 Truncated via traceback budget' if report.raw_log_length > MAX_CHARS else '🔍 Full log inspected'} |",
        ""
    ]

    if error_msg:
        lines.extend([
            "> [!WARNING]",
            f"> AI Triage Engine unavailable: {error_msg}. Displaying structural diagnostic summary below.",
            ""
        ])

    if report.failed_tests:
        lines.extend([
            "### 🎯 Failed Test Breakdown",
            "| Test Name | File Path | Exception Type | Details |",
            "| :--- | :--- | :---: | :--- |"
        ])
        for t in report.failed_tests[:15]:  # limit table rows for readability
            short_msg = t.message.replace("\n", " ").strip()
            if len(short_msg) > 60:
                short_msg = short_msg[:57] + "..."
            line_str = f":L{t.line_number}" if t.line_number else ""
            lines.append(f"| `{t.name}` | `{t.file_path}{line_str}` | `{t.exception_type}` | {short_msg} |")
        if len(report.failed_tests) > 15:
            lines.append(f"| ... and **{len(report.failed_tests) - 15}** more | | | |")
        lines.append("")

    if ai_summary and ai_summary.strip():
        lines.extend([
            "### 🧠 Senior AI Mentor Analysis",
            "",
            ai_summary.strip(),
            ""
        ])

    if suspected_bugs:
        lines.append("### 🐞 Root Cause Hypotheses")
        for bug in suspected_bugs:
            lines.append(f"- {bug}")
        lines.append("")

    if next_steps:
        lines.append("### 🛠️ Minimal Actionable Remediation Steps")
        for step in next_steps:
            lines.append(f"- {step}")
        lines.append("")

    if lesson:
        lines.extend([
            "### 💡 Pedagogical Takeaway",
            lesson.strip(),
            ""
        ])

    lines.extend([
        "---",
        f"*✨ PolyMentor Autonomous CI Engine | Model: `{model_name}` | Triage Time: `{elapsed_ms:.0f}ms`*"
    ])
    return "\n".join(lines)


async def run_pytest_triage(
    log_file: str,
    output_path: str = "pytest_triage_comment.md",
    json_summary: Optional[str] = None,
    repo_root: Optional[str] = None,
    fail_on_error: bool = False,
) -> int:
    out_file = Path(output_path)
    log_p = Path(log_file)
    
    if not log_p.exists():
        msg = f"Log file `{log_file}` not found."
        write_comment(f"## 🧪 PolyMentor Pytest Triage\n\n> [!ERROR]\n> {msg}", out_file)
        return 1 if fail_on_error else 0

    raw_text = log_p.read_text(encoding="utf-8", errors="replace")
    if not raw_text.strip():
        msg = "Pytest execution log was empty."
        write_comment(f"## 🧪 PolyMentor Pytest Triage\n\n> [!NOTE]\n> {msg}", out_file)
        return 0

    # 1. Parse structured metrics
    report = parse_pytest_log(raw_text)
    
    # 2. Extract grounded repository context
    source_context = ground_failure_with_source_code(report, repo_root=repo_root)
    
    # 3. Truncate log intelligently if needed
    trimmed_log, was_truncated = extract_critical_traceback_tail(raw_text, max_chars=MAX_CHARS)
    
    ai_summary = None
    suspected_bugs = []
    next_steps = []
    lesson = None
    model_name = "local-diagnostic"
    elapsed_ms = 0.0
    error_msg = None
    
    if not os.getenv("GROQ_API_KEY"):
        error_msg = "`GROQ_API_KEY` is not configured in environment secrets"
        if fail_on_error:
            print("ERROR: GROQ_API_KEY not set and --fail-on-triage-error specified.", file=sys.stderr)
            return 1
    else:
        print(f"Executing deep AI triage on pytest log ({len(raw_text)} chars, {len(report.failed_tests)} identified test failures)...")
        try:
            pipeline = PolyMentorPipeline.from_groq()
            question = (
                "You are a Principal Software Reliability Engineer and expert CI Triage Mentor. "
                "Analyze this pytest failure report and grounded repository source code. Identify:\n"
                "1. The underlying architectural root cause of the failures/errors.\n"
                "2. Whether this represents a logic regression, a collection/import dependency bug, or a flaky environment issue.\n"
                "3. Concrete, minimal drop-in code fix steps to turn the test suite green.\n"
                "Be decisive, technical, and precise in your guidance."
            )
            
            full_payload = f"=== PYTEST LOG EXECUTION TAIL ===\n{trimmed_log}\n\n=== REPOSITORY SOURCE GROUNDING ===\n{source_context}"
            
            result = await pipeline.analyze(
                code=full_payload,
                language="python",
                level="advanced",
                question=question,
            )
            
            if result.status == "ok":
                ai_summary = result.answer
                suspected_bugs = result.suspected_bugs
                next_steps = result.next_steps
                lesson = result.lesson
                model_name = result.model
                elapsed_ms = result.elapsed_ms
            else:
                error_msg = f"Pipeline returned status: {result.status}"
                if fail_on_error:
                    return 1
        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            print(f"AI triage execution failed: {error_msg}", file=sys.stderr)
            if fail_on_error:
                return 1

    # 4. Generate Enterprise PR Comment
    comment_md = generate_triage_comment_md(
        report=report,
        ai_summary=ai_summary,
        suspected_bugs=suspected_bugs,
        next_steps=next_steps,
        lesson=lesson,
        model_name=model_name,
        elapsed_ms=elapsed_ms,
        error_msg=error_msg
    )
    write_comment(comment_md, out_file)
    
    # 5. Output JSON Summary if requested
    if json_summary:
        json_p = Path(json_summary)
        summary_data = {
            "total_failures": report.total_failures,
            "total_errors": report.total_errors,
            "failed_tests": [
                {"name": f.name, "file": f.file_path, "line": f.line_number, "exception": f.exception_type}
                for f in report.failed_tests
            ],
            "model_name": model_name,
            "elapsed_ms": elapsed_ms,
            "triage_success": error_msg is None,
            "error": error_msg
        }
        json_p.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
        print(f"Wrote machine-readable triage summary to {json_p.absolute()}")

    return 0


async def main() -> None:
    parser = argparse.ArgumentParser(description="PolyMentor Pytest Failure CI Triage Bot")
    parser.add_argument("log_path", help="Path to pytest output console log")
    parser.add_argument("--output", "-o", default="pytest_triage_comment.md", help="Destination markdown comment path")
    parser.add_argument("--json-summary", "-j", default=None, help="Optional JSON diagnostic metrics output file")
    parser.add_argument("--repo-root", "-r", default=None, help="Root path of repository workspace")
    parser.add_argument("--fail-on-triage-error", action="store_true", help="Return non-zero exit code if AI triage pipeline errors out")
    
    args = parser.parse_args()
    exit_code = await run_pytest_triage(
        log_file=args.log_path,
        output_path=args.output,
        json_summary=args.json_summary,
        repo_root=args.repo_root,
        fail_on_error=args.fail_on_triage_error,
    )
    if exit_code != 0:
        sys.exit(exit_code)
    

if __name__ == "__main__":
    asyncio.run(main())

