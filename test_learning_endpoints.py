#!/usr/bin/env python3
"""Smoke tests for the learning guidance endpoints."""

import json

import pytest
import requests

BASE_URL = "http://localhost:8000"


@pytest.mark.skipif(not requests.get(f"{BASE_URL}/health", timeout=2).ok, reason="API server is not running")
def test_learning_endpoints_are_available():
    concepts_response = requests.get(f"{BASE_URL}/learn/concepts", timeout=5)
    assert concepts_response.ok
    concepts = concepts_response.json()
    assert concepts["total_concepts"] > 0

    detail_response = requests.get(f"{BASE_URL}/learn/concept/comparison_operators", timeout=5)
    assert detail_response.ok
    detail = detail_response.json()
    assert detail["concept_name"]

    buggy_code = """for i in range(5):
    if i = 5:
        print(i)"""
    learn_response = requests.post(
        f"{BASE_URL}/learn/from-error",
        json={"code": buggy_code, "language": "python", "level": "beginner"},
        timeout=5,
    )
    assert learn_response.ok
    learn_payload = learn_response.json()
    assert learn_payload["status"] in {"analyzed", "unsupported_language"}

    path_response = requests.get(f"{BASE_URL}/learn/path/comparison_operators", timeout=5)
    assert path_response.ok
    path_payload = path_response.json()
    assert path_payload["starting_concept"]

    explain_response = requests.post(
        f"{BASE_URL}/learn/explain-code",
        json={"code": "for i in range(5):\n    if i == 3:\n        print('Found 3!')", "language": "python", "level": "beginner"},
        timeout=5,
    )
    assert explain_response.ok
    explain_payload = explain_response.json()
    assert explain_payload["status"] in {"analyzed", "unsupported_language"}
