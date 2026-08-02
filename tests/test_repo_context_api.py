"""
tests/test_repo_context_api.py
------------------------------
Integration tests verifying repository workspace root and file path processing
in PolyMentorPipeline and FastAPI /chat endpoint.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from src.inference.pipeline import PolyMentorPipeline, MentorResponse
from src.api.app import app, pipeline as api_pipeline


@pytest.mark.asyncio
async def test_pipeline_repo_auto_extraction(tmp_path):
    # Setup test workspace
    root = tmp_path / "test_repo"
    root.mkdir()
    file_p = root / "calculator.py"
    file_p.write_text("class Calculator:\n    def add(self, a: int, b: int) -> int: return a + b", encoding="utf-8")
    (root / "helpers.py").write_text("def log(): pass", encoding="utf-8")

    pipeline = PolyMentorPipeline(api_key="mock_key_for_test")
    mock_choice = MagicMock()
    mock_choice.message.content = '{"answer": "Looks good!", "suspected_bugs": [], "next_steps": []}'
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]
    
    pipeline._client = MagicMock()
    pipeline._client.chat.completions.create = AsyncMock(return_value=mock_completion)

    res = await pipeline.chat(
        message="Explain this Calculator class",
        code="import helpers\nclass Calculator:\n    def add(self, a, b): return a + b",
        language="python",
        repo_root=str(root),
        file_path="calculator.py",
    )

    assert res.status == "ok"
    call_args = pipeline._client.chat.completions.create.call_args[1]
    sys_prompt = call_args["messages"][0]["content"]
    assert "File: calculator.py" in sys_prompt
    assert "AST Classes: Calculator" in sys_prompt
    assert "AST Functions/Methods: add" in sys_prompt


def test_api_chat_with_repo_coordinates(monkeypatch, tmp_path):
    client = TestClient(app)
    
    mock_res = MentorResponse(
        status="ok",
        answer="Grounded response in repository structure.",
        language="python",
        level="intermediate",
        model="llama-3.3-70b-versatile",
        grounded=True,
    )
    
    captured_kwargs = {}
    async def mock_chat(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return mock_res
        
    monkeypatch.setattr(api_pipeline, "chat", mock_chat)

    payload = {
        "message": "Analyze my dependencies",
        "code": "import requests",
        "language": "python",
        "repo_root": str(tmp_path),
        "file_path": "network/fetch.py",
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["answer"] == "Grounded response in repository structure."
    assert captured_kwargs.get("repo_root") == str(tmp_path)
    assert captured_kwargs.get("file_path") == "network/fetch.py"
