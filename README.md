# PolyMentor

PolyMentor is the AI mentor layer of the **PolyCode ecosystem** (SENODROOM): a Groq-powered coding tutor, a deployable FastAPI service, an MLOps training pipeline, and the documentation site that explains how everything fits together.

**Today:** learners chat in PolyCode; Groq answers in real time; every conversation (and thumbs up/down feedback) is stored in MongoDB.

**Next:** export rated prompts, fine-tune a LoRA adapter on a GPU, evaluate against Groq, and promote only when quality improves.

## What this repository contains

| Piece | Role |
| --- | --- |
| **`src/api/app.py`** | FastAPI service — `/health`, `/chat`, code analysis, learning paths, hints |
| **`src/inference/`** | Groq mentor pipeline + terminal tutor |
| **`src/data_pipeline/`** | MongoDB prompt export and dataset cleaning |
| **`src/training/`** | LoRA fine-tuning (`finetune_chatbot.py`) |
| **`scripts/`** | Export, train, preprocess, local Windows startup |
| **`.github/workflows/`** | Daily deployed API checks + scheduled MongoDB export |
| **`website/`** | React guide site (Setup, PolyCode, MLOps, Deploy, Vision) |

## Related repositories

| Repo | Purpose |
| --- | --- |
| [PolyCode-Frontend](https://github.com/QuantumLogicsLabs/PolyCode-Frontend) | React learner app — lessons, playground, in-app PolyMentor assistant |
| [PolyCode-Backend](https://github.com/QuantumLogicsLabs/PolyCode-Backend) | Express API — auth, chat, profiles, certificates, prompt storage |
| [PolyMentor](https://github.com/QuantumLogicsLabs/PolyMentor) | This repo — FastAPI mentor, training pipeline, automation, docs site |
| [PolyMentor-Website](https://github.com/QuantumLogicsLabs/PolyMentor-Website) | Standalone deployment of the `website/` guide (git submodule) |

## Architecture

```text
PolyCode React app
  -> Express backend (/api/chat/assistant)
       -> Groq API (live answers)
       -> MongoDB polycode.prompts (userMessage, assistantMessage, liked, context)
  -> optional: PolyMentor FastAPI (https://poly-mentor-bm2s.vercel.app)

MLOps loop
  -> GitHub Actions exports + cleans prompts
  -> GPU worker runs LoRA training (scripts/train.sh)
  -> evaluate.sh: blind Groq-judged comparison, LoRA vs Groq, on a fixed eval set
  -> promotes the checkpoint locally if it wins; serving it still needs a
     separate GPU-hosted inference path (manual/future) — Groq stays production

Automation
  -> polymentor-daily.yml — smoke-test deployed /health and /chat
  -> mongodb-prompts-pipeline.yml — export training JSON on schedule
  -> start-polymentor-local.ps1 — start API on Windows laptop
  -> register-polymentor-daily-task.ps1 — Windows Task Scheduler wrapper
```

PolyCode’s in-app assistant uses the **Node backend** by default (`ASSISTANT_PROVIDER=polymentor|groq|custom` in PolyCode Backend). The Python FastAPI service can run separately on Vercel or locally for direct `/chat` access and advanced analysis endpoints.

## Quick start

### 1. Environment

Copy `.env.example` to `.env` and set at least:

```bash
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
```

For training data export, also set `MONGODB_URI` (or `MONGODB_USER`, `MONGODB_PASSWORD`, `MONGODB_CLUSTER`).

### 2. Install (Python 3.12)

```bash
cd PolyMentor
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e .
```

### 3. Run the FastAPI mentor API

```bash
uvicorn src.api.app:app --reload --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs**

**Windows one-liner** (creates `.venv`, installs deps, starts in background):

```powershell
.\scripts\start-polymentor-local.ps1
# Optional: register daily local startup
.\scripts\register-polymentor-daily-task.ps1 -StartTime 09:00 -Port 8000
```

### 4. Run the guide website

```bash
cd website
npm install
npm run dev
```

Open **http://localhost:5173** — routes: `/`, `/setup`, `/polycode`, `/mlops`, `/deploy`, `/vision`.

### 5. Run the full PolyCode stack (what learners use)

```bash
# Terminal 1 — PolyCode Backend
cd PolyCode-Backend
npm run dev

# Terminal 2 — PolyCode Frontend
cd PolyCode-Frontend
npm start
```

Set `GROQ_API_KEY` in the backend `.env`. The PolyMentor FAB in PolyCode posts to `/api/chat/assistant` and saves feedback to MongoDB.

### 6. Terminal tutor (no browser)

```bash
export GROQ_API_KEY=your_key
bash scripts/run_tutor.sh
```

## API overview

Deployed example: **https://poly-mentor-bm2s.vercel.app**

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness check |
| `POST /chat` | Groq-powered mentor chat (`level`: beginner \| intermediate \| advanced) |
| `POST /analyze` | Code analysis with error detection |
| `POST /analyze/detailed` | Deeper analysis |
| `GET /learn/concepts` | List teachable concepts |
| `GET /learn/concept/{id}` | Concept explanation |
| `POST /learn/from-error` | Turn an error into a lesson |
| `GET /learn/path/{id}` | Structured learning path |
| `POST /learn/explain-code` | Explain what code does |
| `GET /learn/hints/{error_type}` | Step-by-step hints by error type |
| `POST /learn/next-hint` | Adaptive next hint |
| `POST /learn/adaptive-level` | Suggest difficulty level |

### Chat example

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "level": "beginner",
    "message": "Teach me Python loops with a small example"
  }'
```

`/chat` response fields include `answer`, `suspected_bugs`, `fixed_code`, `lesson`, `next_steps`, and `elapsed_ms`.

## MLOps — from chats to a better model

### 1. Store conversations in MongoDB

PolyCode Backend writes each assistant turn to `polycode.prompts` with optional `liked` (true/false) from in-app feedback.

### 2. Export training data

```bash
export MONGODB_URI="mongodb+srv://user:pass@cluster/?retryWrites=true&w=majority"
export MONGODB_DB=polycode
export MONGODB_COLLECTION=prompts

python scripts/export_mongodb_prompts.py
# Output: data/processed/mongodb_prompts.json
```

Export only rated conversations:

```bash
python scripts/export_mongodb_prompts.py --only-liked
```

Cleaned record shape:

```json
{
  "userMessage": "How do I fix this loop?",
  "assistantMessage": "You are missing a colon after the for line...",
  "liked": true
}
```

### 3. Preprocess (optional)

```bash
bash scripts/preprocess.sh
```

### 4. Fine-tune on GPU

Groq remains the **production** runtime until a custom checkpoint beats it on eval.

On Windows, use **Python 3.12** and a CUDA PyTorch build for training:

```bash
py -3.12 -m venv venv312
.\venv312\Scripts\Activate.ps1
python -m pip install -e .
python -m pip install -r requirements-train.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

export FETCH_MONGODB_PROMPTS=1
bash scripts/train.sh
```

Default checkpoint:

```text
models_saved/polymentor-chatbot-lora
```

### 5. Evaluate and promote

```bash
bash scripts/evaluate.sh
```

This runs a real comparison, not a manual checklist: it generates answers to a
fixed prompt set (`data/eval/eval_prompts.json`) from both the local LoRA
checkpoint and Groq, then has Groq itself (`GROQ_FALLBACK_MODEL`, blind,
order-randomized) judge each pair. A report is written to
`experiments/logs/eval_<timestamp>.json`. If the LoRA checkpoint's win rate
clears `PROMOTE_THRESHOLD` (default 55%) over at least `PROMOTE_MIN_COMPARISONS`
(default 10) non-tie comparisons, it's copied to
`models_saved/polymentor-chatbot-lora-promoted/` with a promotion manifest.

Promotion only means "this checkpoint is good enough to consider" — it does
**not** switch `/chat` over automatically. Serving a promoted checkpoint in
place of Groq would need a separate, persistent GPU-hosted inference path,
which is a deliberate future step, not something this repo automates today.
Groq remains the production runtime regardless of promotion.

This is designed to run for free: LoRA generation happens locally (same GPU
session you already used to train), and the Groq calls are low-volume (~20
prompts) against the API key you already have.

### GitHub Actions automation

| Workflow | Schedule | Purpose |
| --- | --- | --- |
| `pytest.yml` | On PR, push to `main`, manual | Run unit tests; on PR failure, Groq fail-triage sticky comment |
| `polymentor-daily.yml` | Daily 06:00 UTC | Smoke-test deployed `/health` and `/chat` |
| `mongodb-prompts-pipeline.yml` | Hourly (manual dispatch too) | Export MongoDB prompts → artifact `mongodb_prompts.json` |
| `pr-mentor.yml` | On PR open/sync | Groq PR review sticky comment |

Add repository secrets:

- **`MONGODB_URI`** — required for the export workflow
- **`GROQ_API_KEY`** — required for PR review and pytest fail-triage comments

## Environment variables

| Variable | Required | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | Yes (API/chat) | Groq authentication |
| `GROQ_MODEL` | No | Default chat model (e.g. `llama-3.3-70b-versatile`) |
| `GROQ_FALLBACK_MODEL` | No | Fallback when hybrid/custom routing is enabled |
| `MONGODB_URI` | For export / Actions | Full MongoDB connection string |
| `MONGODB_DB` | No | Database name (default `polycode`) |
| `MONGODB_COLLECTION` | No | Collection name (default `prompts`) |
| `POLYMENTOR_API_URL` | PolyCode Backend | Base URL for `ASSISTANT_PROVIDER=polymentor` (default deployed Vercel URL) |

## Project structure

```text
PolyMentor/
├── src/
│   ├── api/app.py              # FastAPI application (v0.3)
│   ├── inference/              # Groq pipeline, tutor, predict
│   ├── data_pipeline/          # MongoDB export, clean, tokenize
│   ├── training/               # LoRA fine-tuning
│   ├── evaluation/             # Model evaluation helpers
│   ├── analysis/               # Advanced code analyzer
│   ├── learning/               # Concept library and paths
│   ├── reasoning_engine/       # Hints, feedback scoring
│   ├── models/                 # Groq adapter + language heuristics
│   └── utils/
├── scripts/
│   ├── export_mongodb_prompts.py
│   ├── train.sh / preprocess.sh
│   ├── start-polymentor-local.ps1
│   └── register-polymentor-daily-task.ps1
├── .github/workflows/
├── website/                    # React guide (submodule → PolyMentor-Website)
├── tests/
└── docs/
```

## Vision (SENODROOM)

PolyMentor is not only a chatbot. The ecosystem goal is:

1. **Teach with context** — lessons, playground code, and mentor chat share the same learner session.
2. **Turn learning into proof** — profiles, progress, certificates, and follows.
3. **Automate the routine** — GitHub checks, prompt export, scheduled local API startup.
4. **Improve in layers** — Groq today; MongoDB-fed LoRA tomorrow; hybrid inference when eval passes.

See the **Vision** page on the guide site for the full roadmap.

## Development checks

```bash
python -m py_compile src/api/app.py src/inference/pipeline.py src/inference/tutor_mode.py
python -m pytest tests/ -q
npm --prefix website run build
```

## Deployment

- **FastAPI:** Vercel entrypoint `src.api.app:app` (see `pyproject.toml` `[tool.vercel]`)
- **Guide site:** `cd website && npm run build` — deploy `dist/` (Vercel root: `website`)
- **PolyCode:** deploy Frontend + Backend separately; point `POLYMENTOR_API_URL` at your FastAPI host if not using Groq directly

## References

- Groq chat API: https://console.groq.com/docs/text-chat
- Groq API reference: https://console.groq.com/docs/api-reference
- Interactive API docs (local): http://127.0.0.1:8000/docs
