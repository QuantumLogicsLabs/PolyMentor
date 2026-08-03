"""
Tests for PolyMentor CLI analyzer utilities and language inference.
"""

import os
import pytest
from pathlib import Path
from scripts.analyze_file import infer_language_from_filename, find_repo_root, check_quality_gates



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
        isolated_file.write_text("nothing here", encoding="utf-8")
        assert find_repo_root(isolated_file) is None


class DummyResult:
    def __init__(self, status="ok", suspected_bugs=None, static_summary=None):
        self.status = status
        self.suspected_bugs = suspected_bugs or []
        self.static_analysis_summary = static_summary or {}


class TestQualityGates:
    def test_check_quality_gates_passing(self):
        res = DummyResult(status="ok", suspected_bugs=[], static_summary={"supported": True, "quality_score": 95, "total_errors": 0})
        assert check_quality_gates(res, fail_on_bugs=True, min_score=80.0, json_output=True) == 0

    def test_check_quality_gates_failing_bugs(self):
        res = DummyResult(status="ok", suspected_bugs=["Potential null pointer exception"], static_summary={"supported": True, "quality_score": 90, "total_errors": 0})
        assert check_quality_gates(res, fail_on_bugs=True, min_score=80.0, json_output=True) == 2

    def test_check_quality_gates_failing_score(self):
        res = DummyResult(status="ok", suspected_bugs=[], static_summary={"supported": True, "quality_score": 65, "total_errors": 0})
        assert check_quality_gates(res, fail_on_bugs=False, min_score=80.0, json_output=True) == 3

    def test_check_quality_gates_error_status(self):
        res = DummyResult(status="error")
        assert check_quality_gates(res) == 1



class MockPipeline:
    def __init__(self):
        self.calls = []
        
    async def analyze(self, code, language, level, question, repo=None):
        self.calls.append({"code": code, "language": language, "repo": repo})
        from src.inference.pipeline import MentorResult
        return MentorResult(
            answer="Clean code review.",
            status="ok",
            model="mock-model",
            elapsed_ms=10.5,
            grounded=True,
        )


class TestBatchAnalyzer:
    @pytest.mark.asyncio
    async def test_analyze_target_batch(self, tmp_path):
        from scripts.analyze_file import analyze_target
        
        test_dir = tmp_path / "project_source"
        test_dir.mkdir()
        (test_dir / "app.py").write_text("def main(): pass", encoding="utf-8")
        (test_dir / "index.js").write_text("console.log('hello');", encoding="utf-8")
        (test_dir / "ignore.tmp").write_text("ignore me", encoding="utf-8")
        
        mock_pipe = MockPipeline()
        results = await analyze_target(str(test_dir), language="auto", level="advanced", model="mock", json_output=True, pipeline=mock_pipe)
        
        assert len(results) == 2  # Only app.py and index.js should be processed
        assert len(mock_pipe.calls) == 2
        langs_seen = {call["language"] for call in mock_pipe.calls}
        assert langs_seen == {"python", "javascript"}


