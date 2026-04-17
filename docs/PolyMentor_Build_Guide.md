# 🧠 PolyMentor — Step-by-Step Build Guide

> A complete walkthrough for building PolyMentor from scratch: an AI-powered coding mentor that detects errors, teaches concepts, and generates progressive hints across multiple programming languages.

---

## Table of Contents

1. [Prerequisites & Environment Setup](#step-1-prerequisites--environment-setup)
2. [Clone & Install the Project](#step-2-clone--install-the-project)
3. [Configure the Project](#step-3-configure-the-project)
4. [Collect & Prepare Datasets](#step-4-collect--prepare-datasets)
5. [Build the Data Pipeline](#step-5-build-the-data-pipeline)
6. [Implement Feature Extraction](#step-6-implement-feature-extraction)
7. [Build the Error Detection Model](#step-7-build-the-error-detection-model)
8. [Build the Reasoning Engine](#step-8-build-the-reasoning-engine)
9. [Build the Explanation & Hint Models](#step-9-build-the-explanation--hint-models)
10. [Train the Models](#step-10-train-the-models)
11. [Evaluate Model Performance](#step-11-evaluate-model-performance)
12. [Set Up the Inference Pipeline](#step-12-set-up-the-inference-pipeline)
13. [Run the Interactive Tutor](#step-13-run-the-interactive-tutor)
14. [Run Tests](#step-14-run-tests)
15. [Deploy with FastAPI & Docker](#step-15-deploy-with-fastapi--docker)

---

## Step 1: Prerequisites & Environment Setup

Before writing a single line of code, make sure your machine meets these requirements.

### System Requirements

- **OS**: Linux / macOS / Windows (WSL2 recommended)
- **Python**: 3.10 or higher
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: Recommended for training (NVIDIA CUDA-compatible); CPU works for inference only

### Install Python & pip

```bash
# Verify Python version (must be 3.10+)
python3 --version

# Upgrade pip
pip install --upgrade pip
```

### Install CUDA (Optional — for GPU training)

If you have an NVIDIA GPU, install the appropriate CUDA toolkit from [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). Then verify:

```bash
nvidia-smi
```

### Install Tree-sitter CLI (for AST parsing)

```bash
pip install tree-sitter
```

---

## Step 2: Clone & Install the Project

### 2.1 — Clone the Repository

```bash
git clone https://github.com/your-org/polymentor.git
cd polymentor
```

### 2.2 — Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2.3 — Install the Package in Editable Mode

This lets Python resolve internal `src/` imports correctly during development.

```bash
pip install -e .
```

### 2.4 — Verify the Installation

```bash
python -c "from src.inference.pipeline import PolyMentorPipeline; print('✅ Import successful')"
```

---

## Step 3: Configure the Project

PolyMentor uses YAML config files to control all model hyperparameters, training settings, and language-specific tokenizer options.

### 3.1 — Create Your Local Config

```bash
cp configs/model_config.yaml configs/model_config.local.yaml
```

### 3.2 — Edit Model Config

Open `configs/model_config.local.yaml` and adjust:

```yaml
model:
  backbone: "microsoft/codebert-base"
  num_labels: 12 # Number of error types in your taxonomy
  max_seq_length: 512
  dropout: 0.1

training:
  batch_size: 16
  learning_rate: 2e-5
  epochs: 10
  warmup_steps: 500
```

### 3.3 — Edit Language Config

Open `configs/language_config.yaml` and verify tokenizer settings for each supported language: C++, Python, JavaScript, and Java.

---

## Step 4: Collect & Prepare Datasets

PolyMentor is trained on a combination of public and custom datasets.

### 4.1 — Download Public Datasets

| Dataset               | Purpose                                | Where to Get It                                                                |
| --------------------- | -------------------------------------- | ------------------------------------------------------------------------------ |
| CodeNet (IBM)         | 14M multi-language code samples        | [github.com/IBM/Project_CodeNet](https://github.com/IBM/Project_CodeNet)       |
| Stack Overflow Dump   | Error Q&A for explanation training     | [archive.org/details/stackexchange](https://archive.org/details/stackexchange) |
| ManyBugs / IntroClass | Labeled real-world bug datasets        | [repairbenchmarks.cs.umass.edu](https://repairbenchmarks.cs.umass.edu)         |
| LeetCode / HackerRank | Buggy solution pairs for hint training | Custom scrape or existing dumps                                                |

Place all raw data under:

```
data/raw/code_datasets/       ← multi-language code samples
data/raw/error_samples/       ← labeled buggy code examples
data/raw/programming_questions/ ← problem + solution pairs
```

### 4.2 — Define the Error Taxonomy

Edit `data/labels/error_types.json` to define all error categories your classifier will detect:

```json
{
  "syntax_error": 0,
  "logical_error": 1,
  "type_error": 2,
  "off_by_one": 3,
  "infinite_loop": 4,
  "null_reference": 5,
  "division_by_zero": 6,
  "bad_practice": 7,
  "structural_issue": 8
}
```

### 4.3 — Define Difficulty Levels

Edit `data/labels/difficulty_levels.json`:

```json
{
  "beginner": 0,
  "intermediate": 1,
  "advanced": 2
}
```

---

## Step 5: Build the Data Pipeline

The data pipeline cleans raw code, tokenizes it, and splits it into train/val/test sets.

### 5.1 — Implement the Collector (`src/data_pipeline/collector.py`)

Write logic to load files from `data/raw/` and normalize them into a unified schema:

```python
{
  "code": "for i in range(10): ...",
  "language": "python",
  "error_types": ["syntax_error"],
  "difficulty": "beginner",
  "explanation": "You used = instead of == ...",
  "hint": "Step 1: Check your conditional operator..."
}
```

### 5.2 — Implement the Cleaner (`src/data_pipeline/cleaner.py`)

- Remove duplicate code snippets
- Strip comments that leak the answer
- Normalize whitespace and indentation

### 5.3 — Implement the Tokenizer (`src/data_pipeline/tokenizer.py`)

Use the Hugging Face tokenizer tied to your CodeBERT backbone:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

def tokenize(code: str, max_length: int = 512):
    return tokenizer(
        code,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
```

### 5.4 — Build the Dataset (`src/data_pipeline/dataset_builder.py`)

Split the cleaned data into 80/10/10 train/val/test and write to:

```
data/processed/train.json
data/processed/val.json
data/processed/test.json
```

### 5.5 — Run the Preprocessing Script

```bash
bash scripts/preprocess.sh
```

Verify that `data/processed/` now contains populated JSON files.

---

## Step 6: Implement Feature Extraction

### 6.1 — AST Parsing (`src/features/ast_parser.py`)

Use Tree-sitter to parse each code snippet into an Abstract Syntax Tree (AST):

```python
from tree_sitter import Language, Parser

# Build grammars for each language
Language.build_library(
    "build/languages.so",
    ["vendor/tree-sitter-python", "vendor/tree-sitter-javascript", ...]
)

def parse_code(code: str, language: str) -> dict:
    parser = Parser()
    parser.set_language(Language("build/languages.so", language))
    tree = parser.parse(bytes(code, "utf8"))
    return tree.root_node
```

### 6.2 — Code Embeddings (`src/features/code_embeddings.py`)

Generate dense vector representations of code using CodeBERT:

```python
from transformers import AutoModel
import torch

model = AutoModel.from_pretrained("microsoft/codebert-base")

def get_embedding(tokens: dict) -> torch.Tensor:
    with torch.no_grad():
        output = model(**tokens)
    return output.last_hidden_state[:, 0, :]  # CLS token embedding
```

### 6.3 — Syntax Tree Builder (`src/features/syntax_tree_builder.py`)

Convert raw AST nodes into structured representations (node type, depth, children) that can be fed into the model alongside token embeddings.

---

## Step 7: Build the Error Detection Model

### 7.1 — Design the Classifier (`src/models/error_detector.py`)

Build a multi-label classifier on top of CodeBERT. A single snippet can have multiple error types simultaneously.

```python
import torch.nn as nn
from transformers import AutoModel

class ErrorDetector(nn.Module):
    def __init__(self, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("microsoft/codebert-base")
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        output = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(output.last_hidden_state[:, 0, :])
        logits = self.classifier(cls)
        return logits  # Raw logits — apply sigmoid for multi-label
```

### 7.2 — Implement the Model Factory (`src/models/model_factory.py`)

The factory handles loading any saved model checkpoint by name:

```python
def load_model(model_name: str, config: dict):
    if model_name == "error_detector":
        model = ErrorDetector(num_labels=config["num_labels"])
        # load weights if checkpoint exists
    return model
```

---

## Step 8: Build the Reasoning Engine

The reasoning engine is the "brain" of PolyMentor — it maps detected errors to concepts and computes a difficulty score.

### 8.1 — Error Classifier (`src/reasoning_engine/error_classifier.py`)

Map raw model output labels back to human-readable error types and programming concepts:

```python
ERROR_TO_CONCEPT = {
    "syntax_error": "Python Syntax Rules",
    "off_by_one": "Loop Indexing & Boundaries",
    "infinite_loop": "Loop Termination Conditions",
    "type_error": "Data Types & Type Casting",
    ...
}

def classify(label: str) -> dict:
    return {
        "error_type": label,
        "concept_taught": ERROR_TO_CONCEPT.get(label, "General Programming")
    }
```

### 8.2 — Feedback Scorer (`src/reasoning_engine/feedback_scorer.py`)

Score the submitted code for quality (readability, complexity, clean code) on a 0–100 scale using heuristics (line length, nesting depth, variable naming, etc.).

### 8.3 — Hint System (`src/reasoning_engine/hint_system.py`)

Build a rule-based or model-driven step-by-step hint generator. Hints should be progressive — each step reveals slightly more without giving away the answer:

```
Step 1: Think about what operator is used to compare two values.
Step 2: In Python, = assigns a value. What operator checks equality?
Step 3: Replace = with == inside your if condition.
```

---

## Step 9: Build the Explanation & Hint Models

### 9.1 — Explanation Model (`src/models/explanation_model.py`)

Fine-tune a sequence-to-sequence model (CodeT5 or LLaMA) that takes a code snippet + error label and outputs a plain-English explanation:

```
Input:  [CODE] for i in range(10): if i = 5: [ERROR] syntax_error
Output: "You used = which assigns a value. To compare, use == instead."
```

Use the Hugging Face `Seq2SeqTrainer` for fine-tuning. Training data should come from your Stack Overflow and annotated explanation pairs.

### 9.2 — Hint Generator (`src/models/hint_generator.py`)

Either extend the explanation model with a "hint mode" prompt, or train a separate lightweight model on (error_type, difficulty_level) → step-by-step hint pairs.

---

## Step 10: Train the Models

### 10.1 — Implement the Trainer (`src/training/trainer.py`)

Use PyTorch + Hugging Face `Trainer` for the main training loop. Key settings:

- **Loss**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`) for multi-label classification
- **Optimizer**: AdamW
- **Scheduler**: Linear warmup + decay

### 10.2 — Define Loss & Metrics

In `src/training/loss_functions.py`:

```python
import torch.nn as nn
criterion = nn.BCEWithLogitsLoss()
```

In `src/training/metrics.py`, implement F1 score (micro/macro) for multi-label evaluation.

### 10.3 — Launch Training

```bash
bash scripts/train.sh
```

The best checkpoint will be saved to `models_saved/best_mentor_model.pt`.

---

## Step 11: Evaluate Model Performance

### 11.1 — Run Evaluation

```bash
bash scripts/evaluate.sh
```

This runs `src/evaluation/evaluate.py`, which loads the test set and computes:

- F1 Score (micro and macro)
- Precision & Recall per error type
- Learning Effectiveness Score (custom metric from `learning_effectiveness_score.py`)

### 11.2 — Analyze Errors

Review `src/evaluation/error_analysis.py` output to identify which error types the model struggles with most, and collect more targeted training data if needed.

---

## Step 12: Set Up the Inference Pipeline

The inference pipeline is the single public API that combines all components.

### 12.1 — Build the Pipeline (`src/inference/pipeline.py`)

```python
class PolyMentorPipeline:
    @classmethod
    def from_pretrained(cls, model_path: str):
        # Load error detector, explanation model, hint generator
        ...

    def analyze(self, code: str, language: str, level: str) -> MentorResult:
        tokens = tokenize(code)
        error_labels = self.detector.predict(tokens)
        concept = self.reasoning_engine.classify(error_labels)
        explanation = self.explanation_model.generate(code, error_labels)
        hints = self.hint_system.generate(error_labels, level)
        quality = self.scorer.score(code)
        return MentorResult(
            error_type=error_labels,
            explanation=explanation,
            hint=hints,
            concept_taught=concept,
            quality_score=quality
        )
```

### 12.2 — Test the Pipeline Manually

```python
from src.inference.pipeline import PolyMentorPipeline

mentor = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")

result = mentor.analyze("""
for i in range(10):
    print(i)
    if i = 5:
        break
""", language="python", level="beginner")

print(result.error_type)       # SyntaxError: assignment in condition
print(result.explanation)      # "You used = which assigns a value..."
print(result.hint)             # "Step 1: Think about what operator checks equality..."
print(result.concept_taught)   # "Comparison Operators: == vs ="
print(result.quality_score)    # 72 / 100
```

---

## Step 13: Run the Interactive Tutor

The tutor mode (`src/inference/tutor_mode.py`) creates a session-aware interactive loop where the user submits code and PolyMentor responds as a live mentor.

```bash
bash scripts/run_tutor.sh
```

The tutor session:

1. Accepts code input from the user
2. Detects all errors
3. Picks the most important error to teach first
4. Delivers an explanation, then waits
5. Provides progressive hints if the user asks
6. Advances to the next error once resolved

---

## Step 14: Run Tests

Always run the full test suite before committing or deploying.

```bash
# Run all tests
pytest tests/ -v

# Run individual test suites
pytest tests/test_data_pipeline.py
pytest tests/test_model.py
pytest tests/test_explanations.py
pytest tests/test_inference.py

# Lint check
flake8 src/ --max-line-length=100
```

All tests must pass before opening a pull request.

---

## Step 15: Deploy with FastAPI & Docker

### 15.1 — Create the FastAPI App

Create `src/api/app.py`:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from src.inference.pipeline import PolyMentorPipeline

app = FastAPI(title="PolyMentor API")
mentor = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")

class AnalyzeRequest(BaseModel):
    code: str
    language: str
    level: str = "beginner"

@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    result = mentor.analyze(req.code, req.language, req.level)
    return result.dict()
```

### 15.2 — Start the Server Locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Visit `http://localhost:8000/docs` for the auto-generated Swagger UI.

### 15.3 — Dockerize

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t polymentor:latest .
docker run -p 8000:8000 polymentor:latest
```

### 15.4 — Deploy to AWS

Use AWS ECR to push the Docker image, then deploy via ECS or EC2:

```bash
# Authenticate with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <account>.dkr.ecr.us-east-1.amazonaws.com

# Tag and push
docker tag polymentor:latest <account>.dkr.ecr.us-east-1.amazonaws.com/polymentor:latest
docker push <account>.dkr.ecr.us-east-1.amazonaws.com/polymentor:latest
```

---

## ✅ Build Checklist

| Step | Task                                                          | Status |
| ---- | ------------------------------------------------------------- | ------ |
| 1    | Environment setup (Python 3.10+, CUDA, Tree-sitter)           | DONE   |
| 2    | Clone repo and install dependencies                           | DONE   |
| 3    | Configure YAML files                                          | ☐      |
| 4    | Download and organize datasets                                | ☐      |
| 5    | Build data pipeline (collect, clean, tokenize, split)         | ☐      |
| 6    | Implement feature extraction (AST + embeddings)               | ☐      |
| 7    | Build error detection model (CodeBERT multi-label classifier) | ☐      |
| 8    | Build reasoning engine (concept mapping, scoring, hints)      | ☐      |
| 9    | Fine-tune explanation and hint generation models              | ☐      |
| 10   | Train all models                                              | ☐      |
| 11   | Evaluate and analyze model performance                        | ☐      |
| 12   | Build and test inference pipeline                             | ☐      |
| 13   | Run interactive tutor mode                                    | ☐      |
| 14   | Pass full test suite                                          | ☐      |
| 15   | Deploy via FastAPI + Docker (+ AWS)                           | ☐      |

---

<div align="center">

Built with 🎓 by the QuantumLogics team.

_PolyMentor — Don't just fix code. Understand it._

</div>
