# PolyMentor

PolyMentor is a Groq-powered coding tutor chatbot. It teaches programming,
helps write code in multiple languages, reviews snippets, identifies likely
bugs, explains why they happen, and turns fixes into learning guidance.

The new plan is not to train a local CodeBERT-style scoring model by default.
PolyMentor uses Groq for fast LLM responses and keeps the repository focused on
chat, teaching, bug explanation, and multi-language coding help.

## Purpose

PolyMentor should feel like a patient coding mentor:

- Teach programming concepts at beginner, intermediate, or advanced depth.
- Help users write code in Python, JavaScript, TypeScript, Java, C++, C, C#,
  Go, Rust, PHP, Ruby, Swift, Kotlin, SQL, HTML, and CSS.
- Review code and identify likely bugs.
- Explain the reason behind a fix instead of only giving the answer.
- Generate corrected code, examples, and practice tasks.
- Ask clarifying questions when the requested program is underspecified.

## Groq Setup

PolyMentor uses the Groq Python SDK and the Chat Completions API.

```bash
python -m pip install -e .
python -m pip install -r requirements.txt
export GROQ_API_KEY="your_groq_api_key"
```

Optional model override:

```bash
export GROQ_MODEL="llama-3.3-70b-versatile"
```

## Run the Chatbot

Interactive terminal tutor:

```bash
bash scripts/run_tutor.sh
```

FastAPI server:

```bash
uvicorn src.api.app:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Optional Local Fine-Tuning

Groq remains the default runtime. If you want to experiment with a local
PolyMentor LoRA adapter on an NVIDIA GPU such as an RTX 4050, install a CUDA
PyTorch build first. On Windows, use Python 3.12 for CUDA training because
PyTorch CUDA wheels are not available for every newest Python release.

```bash
deactivate  # if another venv is active
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install -r requirements.txt
python -m pip install -r requirements-train.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

Then run:

```bash
bash scripts/train.sh
```

The default checkpoint path is:

```text
models_saved/polymentor-chatbot-lora
```

## API Example

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "language": "python",
    "level": "beginner",
    "message": "Find the bug and teach me the concept",
    "code": "for i in range(10)\n    print(i)"
  }'
```

Response fields include:

- `answer`: the full mentor response
- `suspected_bugs`: extracted likely bug bullets when present
- `fixed_code`: first corrected code block when present
- `lesson`: teaching section when present
- `next_steps`: extracted next-step bullets when present

No numeric quality score is returned.

## Python API

```python
from src.inference.pipeline import PolyMentorPipeline

mentor = PolyMentorPipeline.from_groq()

result = mentor.analyze(
    code="for i in range(10)\n    print(i)",
    language="python",
    level="beginner",
)

print(result.answer)
```

## Project Structure

```text
PolyMentor/
├── src/
│   ├── api/app.py                 # FastAPI chat/review/teach endpoints
│   ├── inference/pipeline.py      # Groq-powered mentor pipeline
│   ├── inference/tutor.py         # Interactive terminal tutor
│   ├── models/groq_model.py       # Groq model adapter
│   ├── models/*_error_detector.py # Optional local language heuristics
│   └── analysis/                  # Optional local code analysis helpers
├── configs/model_config.yaml      # Groq and chatbot configuration
├── scripts/run_tutor.sh           # Start terminal tutor
├── scripts/train.sh               # Explains that local training is not default
├── scripts/evaluate.sh            # Explains conversational evaluation
├── website/                       # React teaching/marketing site
└── docs/                          # Architecture and usage docs
```

## Current Architecture

```text
User question + optional code
        |
        v
PolyMentorPipeline
        |
        v
Groq Chat Completions API
        |
        v
Mentor response:
  - likely bugs
  - explanation
  - fixed code
  - lesson
  - next steps
```

## Environment Variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes | Authenticates calls to Groq. |
| `GROQ_MODEL` | No | Overrides the default Groq chat model. |

## What Changed From The Old Plan

Removed as the default direction:

- Local checkpoint training as the main product path.
- CodeBERT/CodeT5 as required runtime components.
- F1, BERTScore, and numeric quality scoring as user-facing outputs.
- `models_saved/best_mentor_model.pt` as the required entrypoint.

Kept as useful support:

- Multi-language bug-detection ideas.
- AST and analysis helpers for future context enrichment.
- Documentation and examples that help teach programming.

## Development Checks

```bash
python -m py_compile src/inference/pipeline.py src/api/app.py src/inference/tutor_mode.py
npm --prefix website run build
```

## Groq References

Groq official docs describe the Python SDK, `GROQ_API_KEY`, and chat
completion calls through `client.chat.completions.create(...)`.

- https://console.groq.com/docs/text-chat
- https://console.groq.com/docs/api-reference
