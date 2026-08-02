"""
tests/test_context_aware_chat.py
--------------------------------
Unit tests verifying context-aware conversation abilities in PolyMentorPipeline.
Tests code submission, conversational history forwarding, skill level guidance, and language normalization.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.inference.pipeline import PolyMentorPipeline, ChatMessage, MentorResponse
from src.inference.context_builder import RepoContext


@pytest.mark.asyncio
async def test_pipeline_chat_context_assembly():
    pipeline = PolyMentorPipeline(api_key="mock_key_to_avoid_early_exit")
    # Mock groq client response
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "Here is the beginner fix.", "suspected_bugs": ["typo"], "next_steps": ["run pytest"]}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    pipeline._client = MagicMock()
    pipeline._client.chat.completions.create = AsyncMock(return_value=mock_completion)

    history = [
        {"role": "user", "content": "Hello! How do I declare a list?"},
        {"role": "assistant", "content": "You can declare a list using square brackets: my_list = []"},
    ]
    repo = RepoContext(file_path="app.py", dependencies=["pytest"])

    res = await pipeline.chat(
        message="Now how do I append to it?",
        code="my_list = []\nmy_list.add(1)",
        language="py",  # Should normalize to python
        level="beginner",
        history=history,
        repo=repo,
    )

    assert res.status == "ok"
    assert res.language == "python"
    assert res.level == "beginner"
    assert res.answer == "Here is the beginner fix."
    assert "typo" in res.suspected_bugs
    assert res.token_utilization_pct > 0.0

    # Verify what was actually passed to groq client
    call_args = pipeline._client.chat.completions.create.call_args[1]
    messages_passed = call_args["messages"]
    
    # Check system prompt includes beginner guidance and repo context
    assert "Pedagogic Level Guidance (BEGINNER)" in messages_passed[0]["content"]
    assert "Repository Location: File: app.py" in messages_passed[0]["content"]
    
    # Check history turns were included
    assert len(messages_passed) == 4  # System, User turn 1, Assistant turn 1, Current User turn
    assert messages_passed[1]["content"] == "Hello! How do I declare a list?"
    assert "my_list.add(1)" in messages_passed[3]["content"]
