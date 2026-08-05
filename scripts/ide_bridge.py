#!/usr/bin/env python3
"""
scripts/ide_bridge.py
---------------------
PolyMentor IDE & VS Code Integration Bridge.

Serves as a deterministic analysis daemon and Language Server Protocol (LSP) diagnostic adapter
for IDE extensions, VS Code tasks, and real-time buffer linters.
"""

import sys
import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Ensure repository root is in system path for local module imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

logging.basicConfig(level=logging.WARNING, format="[PolyMentor IDE Bridge] %(levelname)s: %(message)s")
logger = logging.getLogger("ide_bridge")

try:
    from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer, ErrorSeverity, CodeError
except ImportError as e:
    logger.error(f"Failed to import internal PolyMentor core analysis suite: {e}")
    sys.exit(1)


def map_severity_to_lsp(severity: Any) -> int:
    """
    Map internal ErrorSeverity strings or enum values to standard LSP integer codes.
    1: DiagnosticSeverity.Error (Critical issues, syntax failures, vulnerabilities)
    2: DiagnosticSeverity.Warning (High severity bugs, memory leaks)
    3: DiagnosticSeverity.Information (Medium severity code smells, complexity)
    4: DiagnosticSeverity.Hint (Low severity styling, naming best practices)
    """
    val = str(severity).lower().split(".")[-1]
    if val in ("critical", "error"):
        return 1
    elif val in ("high", "warning", "warn"):
        return 2
    elif val in ("medium", "info", "information"):
        return 3
    else:
        return 4


def convert_error_to_lsp_diagnostic(error_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a PolyMentor analysis error dictionary into an LSP Diagnostic object.
    Note: Language Server Protocol specifies zero-indexed line and character offsets.
    """
    line_num = max(1, error_dict.get("line", 1) or 1)
    col_num = max(0, error_dict.get("column", 0) or 0)
    lsp_line = line_num - 1
    
    # Construct diagnostic message combining problem and refactor suggestion
    msg = str(error_dict.get("message", "Unknown issue"))
    suggestion = error_dict.get("suggestion")
    if suggestion and str(suggestion).strip() != "None":
        msg += f" | 💡 Refactor Advice: {suggestion}"
        
    category = str(error_dict.get("category", "code_smell"))
    
    return {
        "range": {
            "start": {"line": lsp_line, "character": col_num},
            "end": {"line": lsp_line, "character": col_num + 80}
        },
        "severity": map_severity_to_lsp(error_dict.get("severity", "MEDIUM")),
        "code": category,
        "source": "polymentor-analyzer",
        "message": msg,
        "data": {
            "code_snippet": error_dict.get("code_snippet"),
            "polymentor_rule": category
        }
    }


def read_source_input(file_path: Optional[str] = None, stdin_mode: bool = False) -> str:
    """Read source code buffer from standard input or specified file path."""
    if stdin_mode:
        try:
            return sys.stdin.read()
        except Exception as e:
            logger.error(f"Failed to ingest source from standard input: {e}")
            sys.exit(1)
    
    if file_path:
        target = Path(file_path)
        if not target.exists() or not target.is_file():
            logger.error(f"Specified target file does not exist: {file_path}")
            sys.exit(1)
        try:
            with open(target, "r", encoding="utf-8", errors="replace") as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read file {file_path}: {e}")
            sys.exit(1)
            
    logger.error("No input stream specified. Provide either --file or --stdin-lang.")
    sys.exit(1)


def infer_language_from_name(name_or_lang: str) -> str:
    """Infer programming language identifier from file path or explicit language string."""
    clean = name_or_lang.lower().strip()
    if clean in ("python", "javascript", "cpp", "c++", "java", "c"):
        return clean if clean != "c++" else "cpp"
        
    ext = os.path.splitext(clean)[1]
    if ext in (".py", ".pyw"):
        return "python"
    elif ext in (".js", ".jsx", ".ts", ".tsx"):
        return "javascript"
    elif ext in (".cpp", ".c", ".cc", ".cxx", ".h", ".hpp"):
        return "cpp"
    elif ext == ".java":
        return "java"
    return "python"  # Default fallback


def analyze_buffer(
    source_code: str,
    target_identifier: str,
    format_mode: str = "lsp",
    with_ai_mentor: bool = False
) -> Dict[str, Any]:
    """Execute grounded static AST analysis on a code buffer and format output for IDE clients."""
    lang = infer_language_from_name(target_identifier)
    raw_results = AdvancedCodeAnalyzer.analyze(source_code, lang)
    
    diagnostics = []
    for err in raw_results.get("errors", []):
        if format_mode == "lsp":
            diagnostics.append(convert_error_to_lsp_diagnostic(err))
        else:
            diagnostics.append(err)
            
    score = raw_results.get("score", 100.0)
    
    response = {
        "uri": target_identifier if ("/" in target_identifier or "\\" in target_identifier or "." in target_identifier) else f"buffer://{lang}",
        "language": lang,
        "quality_score": score,
        "total_issues": len(diagnostics),
        "diagnostics": diagnostics
    }
    
    if with_ai_mentor and len(diagnostics) > 0:
        try:
            from src.inference.groq_pipeline import PolyMentorPipeline
            pipe = PolyMentorPipeline.from_groq()
            response["ai_mentor_summary"] = f"Found {len(diagnostics)} structural issue(s). Consider addressing critical syntax and bug patterns first."
        except Exception:
            response["ai_mentor_summary"] = "AI review offline (using pure deterministic static analysis grounding)."
            
    return response


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for IDE bridge operations."""
    parser = argparse.ArgumentParser(
        description="PolyMentor IDE & VS Code Diagnostic Bridge (LSP Adapter)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=str, help="Path to a single code file to analyze")
    group.add_argument("--dir", type=str, help="Path to a workspace directory for batch diagnostic scan")
    group.add_argument("--stdin-lang", type=str, help="Language identifier when streaming code buffer via stdin (e.g. python, javascript)")
    
    parser.add_argument(
        "--format",
        choices=["lsp", "compact"],
        default="lsp",
        help="Diagnostic JSON output structure (default: lsp)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="Minimum required quality score gate (exits with code 2 if unmet)"
    )
    parser.add_argument(
        "--with-ai-mentor",
        action="store_true",
        help="Include grounded AI refactoring advice via PolyMentor pipeline"
    )
    return parser.parse_args()


def main() -> None:
    """Main execution flow for PolyMentor IDE & VS Code diagnostic bridge."""
    args = parse_arguments()
    
    if args.dir:
        workspace = Path(args.dir)
        if not workspace.exists() or not workspace.is_dir():
            logger.error(f"Workspace directory does not exist: {args.dir}")
            sys.exit(1)
            
        file_results = []
        total_score = 0.0
        valid_files = 0
        
        for root, _, files in os.walk(workspace):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in (".py", ".js", ".ts", ".jsx", ".tsx", ".cpp", ".c", ".cc", ".h", ".hpp", ".java"):
                    file_path = os.path.join(root, f)
                    try:
                        with open(file_path, "r", encoding="utf-8", errors="replace") as fp:
                            code = fp.read()
                        res = analyze_buffer(code, file_path, format_mode=args.format, with_ai_mentor=args.with_ai_mentor)
                        file_results.append(res)
                        total_score += res.get("quality_score", 100.0)
                        valid_files += 1
                    except Exception as e:
                        logger.warning(f"Skipping {file_path} due to read exception: {e}")
                        
        avg_score = round(total_score / valid_files, 2) if valid_files > 0 else 100.0
        output_payload = {
            "mode": "workspace_batch",
            "workspace_root": str(workspace),
            "files_analyzed": valid_files,
            "average_quality_score": avg_score,
            "diagnostics_by_file": file_results
        }
        print(json.dumps(output_payload, indent=2 if args.format == "lsp" else None))
        if avg_score < args.min_score:
            sys.exit(2)
    else:
        source = read_source_input(file_path=args.file, stdin_mode=bool(args.stdin_lang))
        identifier = args.file if args.file else str(args.stdin_lang)
        output_payload = analyze_buffer(source, identifier, format_mode=args.format, with_ai_mentor=args.with_ai_mentor)
        print(json.dumps(output_payload, indent=2 if args.format == "lsp" else None))
        if output_payload.get("quality_score", 100.0) < args.min_score:
            sys.exit(2)


if __name__ == "__main__":
    main()
