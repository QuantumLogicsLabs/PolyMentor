"""
tests/test_repo_parser.py
-------------------------
Unit tests for PolyMentor Tree-sitter source parsing, AST symbol extraction,
heuristic fallbacks, and repository directory scanning.
"""

import os
import shutil
import tempfile
from pathlib import Path
import pytest
from src.inference.repo_parser import RepoParser
from src.inference.context_builder import RepoContext, ContextBuilder


def test_normalize_lang():
    parser = RepoParser()
    assert parser.normalize_lang("py") == "python"
    assert parser.normalize_lang("PYTHON3") == "python"
    assert parser.normalize_lang("js") == "javascript"
    assert parser.normalize_lang("TypeScript") == "javascript"
    assert parser.normalize_lang("c++") == "cpp"
    assert parser.normalize_lang("java") == "java"


def test_extract_symbols_python():
    parser = RepoParser()
    code = """import os
from pathlib import Path

class UserProcessor:
    def __init__(self, name: str):
        self.name = name

    def process_data(self, x: int) -> bool:
        return x > 0
"""
    symbols = parser.extract_symbols(code, "python")
    assert "UserProcessor" in symbols["classes"]
    assert "init" in symbols["functions"] or "__init__" in symbols["functions"]
    assert "process_data" in symbols["functions"]
    assert any("import os" in imp or "Path" in imp for imp in symbols["imports"])


def test_heuristic_fallback():
    parser = RepoParser()
    code = """class SimpleService:
    def execute(self):
        pass
"""
    # Test directly via fallback
    symbols = parser._extract_heuristic_symbols(code, "python")
    assert "SimpleService" in symbols["classes"]
    assert "execute" in symbols["functions"]


def test_scan_workspace(tmp_path):
    parser = RepoParser()
    root = tmp_path / "my_project"
    root.mkdir()
    
    # Create code files
    (root / "main.py").write_text("print('hello')", encoding="utf-8")
    src_dir = root / "src"
    src_dir.mkdir()
    (src_dir / "utils.py").write_text("def helper(): pass", encoding="utf-8")
    (src_dir / "data.json").write_text("{}", encoding="utf-8")  # Non-code should be ignored
    
    # Create ignored directories
    venv_dir = root / ".venv"
    venv_dir.mkdir()
    (venv_dir / "ignored.py").write_text("pass", encoding="utf-8")
    
    files = parser.scan_workspace(root)
    assert "main.py" in files
    assert "src/utils.py" in files
    assert "src/data.json" not in files
    assert not any(".venv" in f for f in files)


def test_extract_repo_context(tmp_path):
    parser = RepoParser()
    code = "import math\nclass Vector:\n    def norm(self): pass"
    repo_ctx = parser.extract_repo_context(code, "python", root_dir=str(tmp_path), file_path="math_utils.py")
    assert isinstance(repo_ctx, RepoContext)
    assert repo_ctx.file_path == "math_utils.py"
    assert "Vector" in repo_ctx.classes
    assert "norm" in repo_ctx.functions
    
    # Format repo context in ContextBuilder
    formatted = ContextBuilder.format_repo_context(repo_ctx)
    assert "File: math_utils.py" in formatted
    assert "AST Classes: Vector" in formatted
    assert "AST Functions/Methods: norm" in formatted
