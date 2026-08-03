"""
Tests for PolyMentor CLI analyzer utilities and language inference.
"""

import os
import pytest
from pathlib import Path
from scripts.analyze_file import infer_language_from_filename, find_repo_root


class TestCLIAnalyzerUtilities:
    def test_infer_language_from_filename(self):
        assert infer_language_from_filename("test.py") == "python"
        assert infer_language_from_filename("app.ts") == "typescript"
        assert infer_language_from_filename("main.cpp") == "cpp"
        assert infer_language_from_filename("script.unknown") == "python"  # default fallback
        assert infer_language_from_filename("dir/sub/component.jsx") == "javascript"
        assert infer_language_from_filename("style.css") == "css"

    def test_find_repo_root(self, tmp_path):
        repo_dir = tmp_path / "my_project"
        repo_dir.mkdir()
        git_dir = repo_dir / ".git"
        git_dir.mkdir()

        sub_dir = repo_dir / "src" / "components"
        sub_dir.mkdir(parents=True)
        test_file = sub_dir / "app.py"
        test_file.write_text("print('hello')", encoding="utf-8")

        found_root = find_repo_root(test_file)
        assert found_root == str(repo_dir)

        # Test project config fallback (pyproject.toml)
        py_project_dir = tmp_path / "py_proj"
        py_project_dir.mkdir()
        (py_project_dir / "pyproject.toml").write_text("[tool.poetry]", encoding="utf-8")
        py_file = py_project_dir / "main.py"
        py_file.write_text("x = 1", encoding="utf-8")
        assert find_repo_root(py_file) == str(py_project_dir)

        # Test null on isolated folder without repository indicators
        no_git_dir = tmp_path / "isolated"
        no_git_dir.mkdir()
        isolated_file = no_git_dir / "test.txt"
        isolated_file.write_text("nothing here", encoding="utf-8")
        assert find_repo_root(isolated_file) is None
