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


if __name__ == "__main__":
    args = parse_arguments()
    # Foundational entry point; subsequent steps integrate diagnostic formatting and AST pipeline invocation
    print(json.dumps({"status": "initialized", "mode": args.format}))
