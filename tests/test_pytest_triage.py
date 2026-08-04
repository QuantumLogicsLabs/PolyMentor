"""
tests/test_pytest_triage.py
---------------------------
Unit test suite verifying structured pytest log parsing, critical traceback tail trimming,
repository source code grounding, diagnosis scorecard generation, and CLI automation.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch
import pytest

from scripts.triage_pytest_failure import (
    FailedTest,
    PytestReport,
    extract_critical_traceback_tail,
    generate_triage_comment_md,
    ground_failure_with_source_code,
    parse_pytest_log,
    run_pytest_triage,
)

SAMPLE_PYTEST_LOG = """
==================================== ERRORS ====================================
______________ ERROR collecting tests/test_context_aware_chat.py _______________
ImportError while importing test module '/home/runner/work/PolyMentor/PolyMentor/tests/test_context_aware_chat.py'.
E ModuleNotFoundError: No module named 'slowapi'
=================================== FAILURES ===================================
_________________ TestBatchAnalyzer.test_analyze_target_batch __________________
tests/test_cli_analyzer.py:106: in test_analyze_target_batch
    assert len(results) == 2
E   assert 0 == 2
=========================== short test summary info ============================
ERROR tests/test_context_aware_chat.py - ModuleNotFoundError: No module named 'slowapi'
FAILED tests/test_cli_analyzer.py::test_analyze_target_batch - AssertionError
================== 1 failed, 10 passed, 1 error in 2.34s ==================
"""


def test_parse_pytest_log():
    report = parse_pytest_log(SAMPLE_PYTEST_LOG)
    assert report.total_errors == 1
    assert report.total_failures == 1
    assert len(report.failed_tests) == 2
    
    err_test = report.failed_tests[0]
    assert "test_context_aware_chat.py" in err_test.file_path
    assert err_test.exception_type == "ModuleNotFoundError"
    
    fail_test = report.failed_tests[1]
    assert fail_test.name == "test_analyze_target_batch"
    assert "test_cli_analyzer.py" in fail_test.file_path


def test_extract_critical_traceback_tail_under_budget():
    log, truncated = extract_critical_traceback_tail(SAMPLE_PYTEST_LOG, max_chars=10000)
    assert not truncated
    assert log == SAMPLE_PYTEST_LOG


def test_extract_critical_traceback_tail_over_budget():
    large_log = "HEADER LINE\n" * 60 + "NOISE LINE\n" * 5000 + "\nCRITICAL TAIL ERROR\n"
    log, truncated = extract_critical_traceback_tail(large_log, max_chars=1500)
    assert truncated
    assert "HEADER LINE" in log
    assert "CRITICAL TAIL ERROR" in log
    assert "omitted to preserve critical test failure tracebacks" in log
    assert len(log) <= 1700


def test_ground_failure_with_source_code():
    with tempfile.TemporaryDirectory() as tmp_dir:
        root_path = Path(tmp_dir)
        test_dir = root_path / "tests"
        test_dir.mkdir(exist_ok=True)
        
        target_file = test_dir / "test_sample.py"
        lines = [f"line {i}" for i in range(1, 50)]
        lines[19] = "    assert 1 == 2  # line 20 failing assertion"
        target_file.write_text("\n".join(lines), encoding="utf-8")
        
        report = PytestReport(
            total_failures=1,
            failed_tests=[
                FailedTest(
                    name="test_dummy",
                    file_path="tests/test_sample.py",
                    line_number=20,
                    exception_type="AssertionError",
                    message="assert 1 == 2"
                )
            ]
        )
        
        grounding_out = ground_failure_with_source_code(report, repo_root=str(root_path))
        assert "Repository Source Code Grounding" in grounding_out
        assert "tests/test_sample.py" in grounding_out
        assert "failing assertion" in grounding_out
        assert "--> 20:" in grounding_out
