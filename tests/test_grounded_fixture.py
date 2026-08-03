"""CI gate: analyzer must hit expected issues on the grounded-chat fixture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer
from src.inference.pipeline import PolyMentorPipeline


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "eval"
    / "grounded_chat_snippets.json"
)
MIN_HIT_RATE = 0.8


def _analyzer_hit(snippet: dict, analysis: dict) -> bool:
    expected_cats = set(snippet.get("expected_categories") or [])
    expected_keywords = [k.lower() for k in (snippet.get("expected_keywords") or [])]
    errors = analysis.get("errors") or []
    cats = {e.get("category") for e in errors}
    if expected_cats and cats & expected_cats:
        return True
    blob = " ".join(
        f"{e.get('message') or ''} {e.get('suggestion') or ''}" for e in errors
    ).lower()
    return any(keyword in blob for keyword in expected_keywords)


def test_grounded_fixture_analyzer_hit_rate():
    snippets = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert len(snippets) >= 15, "Fixture should have a meaningful sample size"

    hits = 0
    misses = []
    for snippet in snippets:
        analysis = AdvancedCodeAnalyzer.analyze(snippet["code"], snippet["language"])
        if _analyzer_hit(snippet, analysis):
            hits += 1
        else:
            misses.append(snippet["id"])

    hit_rate = hits / len(snippets)
    assert hit_rate >= MIN_HIT_RATE, (
        f"Analyzer hit-rate {hit_rate:.1%} below {MIN_HIT_RATE:.0%}; "
        f"misses={misses}"
    )


@pytest.mark.asyncio
async def test_grounded_fixture_pipeline_prompt_injection():
    snippets = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    pipeline = PolyMentorPipeline(api_key="mock_key_to_test")
    
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "Here is the grounded advice.", "suspected_bugs": ["bug"]}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    pipeline._client = MagicMock()
    pipeline._client.chat.completions.create = AsyncMock(return_value=mock_completion)

    # Test top 5 snippets across languages to verify hybrid grounding integration
    for snippet in snippets[:5]:
        res = await pipeline.chat(
            message=snippet["question"],
            code=snippet["code"],
            language=snippet["language"],
            level="intermediate",
        )
        assert res.status == "ok"
        assert res.grounded is True
        assert res.static_analysis_summary is not None
        
        # Check system prompt actually received deterministic grounding instructions
        call_args = pipeline._client.chat.completions.create.call_args[1]
        sys_prompt = call_args["messages"][0]["content"]
        assert "Static Analysis Grounding:" in sys_prompt

