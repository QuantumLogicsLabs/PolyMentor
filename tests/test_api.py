#!/usr/bin/env python3
"""
Test script for PolyMentor API endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Testing PolyMentor API")
print("=" * 60)

# Test 1: Health check
print("\n1. Testing /health endpoint...")
try:
    response = requests.get(f"{BASE_URL}/health")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Root endpoint
print("\n2. Testing / endpoint...")
try:
    response = requests.get(f"{BASE_URL}/")
    print(f"   Status: {response.status_code}")
    print(f"   Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Analyze endpoint with syntax error
print("\n3. Testing /analyze with syntax error (assignment in condition)...")
try:
    payload = {
        "code": "for i in range(10):\n    if i = 5:\n        break",
        "language": "python",
        "level": "beginner",
        "num_hints": 3
    }
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Status: {result.get('status')}")
    print(f"   Mode: {result.get('mode')}")
    print(f"   Error Type: {result.get('error_type')}")
    print(f"   Error Types: {result.get('error_types')}")
    print(f"   Explanation: {result.get('explanation')[:100]}...")
    print(f"   Quality Score: {result.get('quality_score')}")
    print(f"   Elapsed (ms): {result.get('elapsed_ms')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: Analyze endpoint with clean code
print("\n4. Testing /analyze with clean code...")
try:
    payload = {
        "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "language": "python",
        "level": "intermediate"
    }
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Status: {result.get('status')}")
    print(f"   Mode: {result.get('mode')}")
    print(f"   Error Type: {result.get('error_type')}")
    print(f"   Quality Score: {result.get('quality_score')}")
    print(f"   Suggestions: {result.get('suggestions')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 5: Analyze endpoint with JavaScript
print("\n5. Testing /analyze with JavaScript code...")
try:
    payload = {
        "code": "for(let i=0; i<10; i++) {\n    if(i = 5) {\n        break;\n    }\n}",
        "language": "javascript",
        "level": "beginner"
    }
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Status: {result.get('status')}")
    print(f"   Language: {result.get('language')}")
    print(f"   Quality Score: {result.get('quality_score')}")
except Exception as e:
    print(f"   Error: {e}")

# Test 6: Empty code
print("\n6. Testing /analyze with empty code...")
try:
    payload = {
        "code": "",
        "language": "python",
        "level": "beginner"
    }
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print(f"   Status: {response.status_code}")
    result = response.json()
    print(f"   Status: {result.get('status')}")
    print(f"   Explanation: {result.get('explanation')}")
except Exception as e:
    print(f"   Error: {e}")

print("\n" + "=" * 60)
print("API tests completed!")
print("=" * 60)
