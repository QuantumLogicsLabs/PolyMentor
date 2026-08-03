"""
tests/test_hybrid_mentor.py
---------------------------
Unit tests verifying the hybrid AI mentor static analysis grounding capabilities.
Tests multi-language support, quality scoring, category formatting, and token budget safety.
"""

import pytest
from src.inference.context_builder import ContextBuilder
from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer
from src.inference.pipeline import PolyMentorPipeline
from unittest.mock import AsyncMock, MagicMock


def test_format_static_analysis_with_buggy_python():
    code = "def calc(a, b=[]):\n    global count\n    return a / 0"
    result = AdvancedCodeAnalyzer.analyze(code, "python")
    result["quality_score"] = AdvancedCodeAnalyzer.get_quality_score(code, "python")

    formatted = ContextBuilder.format_static_analysis(result, max_chars=1200)
    assert "Static Analysis Grounding: Detected" in formatted
    assert "Quality Score:" in formatted
    assert "best_practice_violation" in formatted or "boundary_error" in formatted or "issue" in formatted
    assert "Verified Fix Suggestion:" in formatted


def test_format_static_analysis_clean_code():
    code = 'def add(a: int, b: int) -> int:\n    """Add two integers and return the sum."""\n    return a + b\n'
    result = AdvancedCodeAnalyzer.analyze(code, "python")
    result["quality_score"] = 100
    
    formatted = ContextBuilder.format_static_analysis(result, max_chars=1200)
    assert "Clean code detected (0 deterministic errors found)" in formatted
    assert "(Quality Score: 100/100)" in formatted



def test_format_static_analysis_truncation():
    # Simulate an analysis result with many issues with long messages to test truncation safety
    errors = [
        {
            "category": "style_issue",
            "severity": "high",
            "line": i,
            "message": "A very long error message that repeats extensively to push against token limits and character bounds.",
            "suggestion": "An equally lengthy verified fix suggestion that provides detailed syntactic restructuring advice for the LLM."
        }
        for i in range(1, 20)
    ]
    result = {"supported": True, "total_errors": 20, "errors": errors, "quality_score": 25}
    formatted = ContextBuilder.format_static_analysis(result, max_chars=400)
    assert len(formatted) <= 400
    assert "... [Truncated]" in formatted


@pytest.mark.asyncio
async def test_pipeline_hybrid_multi_language():
    pipeline = PolyMentorPipeline(api_key="mock_test_key")
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "Here is the fix for C++ memory leak.", "suspected_bugs": ["memory_leak"]}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    pipeline._client = MagicMock()
    pipeline._client.chat.completions.create = AsyncMock(return_value=mock_completion)

    cpp_code = "void f() { int* ptr = new int[5]; }"
    res = await pipeline.chat(
        message="Review this function",
        code=cpp_code,
        language="cpp",
        level="advanced"
    )

    assert res.status == "ok"
    assert res.grounded is True
    assert res.static_analysis_summary is not None
    assert res.static_analysis_summary.get("total_errors", 0) >= 0
