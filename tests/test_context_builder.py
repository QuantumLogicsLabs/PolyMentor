"""
tests/test_context_builder.py
-----------------------------
Unit tests for the Intelligent Context Builder (`ContextBuilder`) in PolyMentor.
Verifies token budgeting, truncation algorithms, repository metadata synthesis, and prompt assembly.
"""

import pytest
from src.inference.context_builder import (
    ContextBuilder,
    RepoContext,
    PackedPrompt,
    DEFAULT_TOKEN_BUDGET,
)


def test_estimate_tokens_basic():
    text = "Hello, world! This is a test of token estimation."
    tokens = ContextBuilder.estimate_tokens(text)
    assert tokens > 0
    assert ContextBuilder.estimate_tokens("") == 0


def test_truncate_code_to_budget():
    short_code = "print('Hello world')"
    processed, truncated = ContextBuilder.truncate_code_to_budget(short_code, token_budget=100)
    assert not truncated
    assert processed == short_code

    long_code = "x = 1\n" * 2000  # Highly repetitive long code
    processed_long, truncated_long = ContextBuilder.truncate_code_to_budget(long_code, token_budget=100)
    assert truncated_long
    assert "[Code middle-truncated" in processed_long


def test_format_static_analysis():
    mock_analysis = {
        "supported": True,
        "total_errors": 2,
        "errors": [
            {"line": 10, "severity": "high", "message": "Undefined variable 'foo'"},
            {"line": 15, "severity": "medium", "message": "Unused import 'os'"},
        ],
    }
    summary = ContextBuilder.format_static_analysis(mock_analysis)
    assert "Detected 2 potential issue(s)" in summary
    assert "[Line 10] Undefined variable 'foo'" in summary


def test_pack_history_budget():
    history = [
        {"role": "user", "content": "Question 1 " * 50},
        {"role": "assistant", "content": "Answer 1 " * 50},
        {"role": "user", "content": "Question 2 " * 50},
    ]
    packed, tokens, dropped = ContextBuilder.pack_history(history, max_tokens=100, max_turns=5)
    # Should drop oldest turns when budget is very restricted
    assert dropped > 0
    assert tokens <= 120


def test_build_prompt_integration():
    builder = ContextBuilder(max_tokens=2500)
    repo = RepoContext(file_path="src/main.py", dependencies=["fastapi", "groq"])
    analysis = {"supported": True, "total_errors": 0, "errors": []}

    prompt = builder.build_prompt(
        message="Explain how to set up FastAPI routing.",
        code="app = FastAPI()",
        language="python",
        level="intermediate",
        analysis_result=analysis,
        repo=repo,
    )

    assert isinstance(prompt, PackedPrompt)
    assert prompt.estimated_tokens <= 2500
    assert prompt.grounding_enabled
    assert len(prompt.messages) >= 2
    assert "src/main.py" in prompt.messages[0]["content"]
