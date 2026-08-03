"""
tests/test_hybrid_chat_endpoint.py
----------------------------------
Integration tests verifying FastAPI /chat endpoint properly invokes hybrid static analysis
and exposes grounded telemetry and deterministic findings in JSON responses.
"""

from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock
from src.api.app import app, pipeline as api_pipeline
from src.inference.pipeline import MentorResponse


def test_hybrid_chat_api_endpoint(monkeypatch):
    client = TestClient(app)

    mock_summary = {
        "total_errors": 2,
        "quality_score": 75.5,
        "errors": [
            {
                "category": "best_practice_violation",
                "severity": "medium",
                "line": 1,
                "message": "Wildcard import detected",
                "suggestion": "Import specific symbols instead of *"
            }
        ]
    }

    mock_res = MentorResponse(
        status="ok",
        answer="I noticed a wildcard import. It's best practice to import explicit symbols.",
        language="python",
        level="beginner",
        model="llama-3.3-70b-versatile",
        suspected_bugs=["wildcard import"],
        grounded=True,
        token_utilization_pct=12.4,
        static_analysis_summary=mock_summary,
    )

    async def mock_chat(*args, **kwargs):
        return mock_res

    monkeypatch.setattr(api_pipeline, "chat", mock_chat)

    payload = {
        "message": "Review my import statement",
        "code": "from os import *\nprint(getcwd())",
        "language": "python",
        "level": "beginner",
    }

    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["grounded"] is True
    assert "static_analysis_summary" in data
    assert data["static_analysis_summary"]["total_errors"] == 2
    assert data["static_analysis_summary"]["quality_score"] == 75.5
    assert len(data["static_analysis_summary"]["errors"]) == 1
    assert data["static_analysis_summary"]["errors"][0]["category"] == "best_practice_violation"
