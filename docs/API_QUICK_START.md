# PolyMentor API Quick Start

PolyMentor exposes a Groq-powered coding tutor API.

## Setup

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key"
```

## Start Server

```bash
uvicorn src.api.app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

## Chat

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Teach me Python loops with a small example",
    "language": "python",
    "level": "beginner"
  }'
```

## Review Code

```bash
curl -X POST http://127.0.0.1:8000/review \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Find the bug and explain the fix",
    "language": "python",
    "level": "beginner",
    "code": "for i in range(10)\n    print(i)"
  }'
```

## Teach A Topic

```bash
curl -X POST http://127.0.0.1:8000/teach \
  -H "Content-Type: application/json" \
  -d '{
    "message": "JavaScript promises",
    "language": "javascript",
    "level": "intermediate"
  }'
```

## Response Shape

```json
{
  "status": "ok",
  "answer": "...",
  "language": "python",
  "level": "beginner",
  "model": "llama-3.3-70b-versatile",
  "suspected_bugs": [],
  "fixed_code": null,
  "lesson": null,
  "next_steps": [],
  "elapsed_ms": 1234.0
}
```

There is no quality-score endpoint in the new plan.
