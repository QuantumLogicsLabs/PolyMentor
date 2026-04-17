<div align="center">

# 🧠 PolyMentor

### AI-Powered Coding Mentor

_Multi-language error detection, concept teaching, and intelligent hints — built for learners who want to understand code, not just fix it._

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-FFD21F?logo=huggingface)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active%20Development-orange)]()

</div>

---

## 📖 Overview

**PolyMentor** is an AI-driven coding mentor that goes beyond error detection. It analyzes source code across multiple programming languages, identifies syntax errors, logical mistakes, and inefficient patterns, then explains them the way a human teacher would — with reasoning, concept teaching, and progressive hints.

Where traditional linters tell you _what_ is wrong, PolyMentor tells you _why_ it is wrong and _how to think about it correctly_. It is designed as a modular ML system that can operate standalone or plug directly into the [Polycode](https://github.com/your-org/polycode) platform as its AI learning layer.

> **QuantumLogics Project** — Sister project to [PolyGuard](https://github.com/your-org/polyguard). Together they form a full AI developer intelligence system.

---

## 🎯 Objectives

1. **Code Understanding & Error Detection** — Analyze multi-language code to detect syntax errors, logical mistakes, and bad coding practices in real time using AST parsing and ML classifiers.

2. **AI-Powered Learning Guidance** — Provide simple, human-like explanations of errors and teach the concepts behind them, not just flag the line number.

3. **Smart Hint & Skill Development System** — Generate step-by-step hints and adaptive guidance calibrated to the learner's level to improve problem-solving skills without giving away answers.

4. **Multi-Language Intelligent Tutor** — Act as a unified AI mentor for C++, Python, JavaScript, and Java, helping users learn, debug, and improve code quality consistently across languages.

---

## ✨ Features

| Feature                          | Description                                                       |
| -------------------------------- | ----------------------------------------------------------------- |
| 🔍 **Real-Time Error Detection** | Syntax errors, logical bugs, off-by-one errors, misused patterns  |
| 🧾 **Human-Like Explanations**   | Explains _why_ the error happened, not just where                 |
| 💡 **Progressive Hint System**   | Step-by-step hints that guide without spoiling the solution       |
| 🎓 **Adaptive Learning Mode**    | Adjusts explanation depth to beginner, intermediate, or advanced  |
| 📊 **Code Quality Scoring**      | Readability, complexity, and clean code improvement suggestions   |
| 🧠 **Concept Teaching**          | Maps errors to programming concepts — loops, recursion, OOP, etc. |
| 🌍 **Multi-Language Support**    | C++, Python, JavaScript · Java coming soon                        |
| 🔗 **Integration-Ready**         | Clean inference API for plug-in to Polycode or any platform       |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    Input: Source Code                         │
│              (C++ / Python / JavaScript / Java)               │
└──────────────────────────┬───────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │   AST + Tokenization    │
              │  Tree-sitter parsing    │
              │  Code embeddings        │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Error Detection Model │
              │   CodeBERT classifier   │
              │   Multi-label output    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │   Reasoning Engine      │
              │  Error classification   │
              │  Concept mapping        │
              │  Difficulty scoring     │
              └────────────┬────────────┘
                           │
         ┌─────────────────┼──────────────────┐
         │                 │                  │
┌────────▼───────┐ ┌───────▼──────┐ ┌────────▼────────┐
│  Explanation   │ │     Hint     │ │  Quality Score  │
│  Generator     │ │   System     │ │  + Suggestions  │
│  (LLM / FT)    │ │  Step-by-step│ │                 │
└────────┬───────┘ └───────┬──────┘ └────────┬────────┘
         └─────────────────┼──────────────────┘
                           │
              ┌────────────▼────────────┐
              │     Final Output        │
              │  · Error type & location│
              │  · Why it happened      │
              │  · Step-by-step hint    │
              │  · Concept taught       │
              │  · Quality score        │
              └─────────────────────────┘
```

The sole public entrypoint is `src/inference/pipeline.py`. All training, feature extraction, and reasoning internals are encapsulated.

---

## 🗂️ Project Structure

```
PolyMentor/
├── configs/
│   ├── model_config.yaml          # Model hyperparameters
│   ├── training_config.yaml       # Training schedule, batch size, LR
│   └── language_config.yaml       # Per-language tokenizer settings
│
├── data/
│   ├── raw/
│   │   ├── code_datasets/         # Collected multi-language code samples
│   │   ├── error_samples/         # Labeled buggy code examples
│   │   └── programming_questions/ # Problem + solution pairs
│   ├── processed/
│   │   ├── train.json
│   │   ├── val.json
│   │   └── test.json
│   └── labels/
│       ├── error_types.json       # Error taxonomy
│       └── difficulty_levels.json # Skill level definitions
│
├── notebooks/
│   ├── data_analysis.ipynb
│   ├── baseline_model.ipynb
│   └── explanation_model_tests.ipynb
│
├── src/
│   ├── data_pipeline/
│   │   ├── collector.py           # Scraping + dataset collection
│   │   ├── cleaner.py             # Noise removal, normalization
│   │   ├── tokenizer.py           # Language-aware tokenization
│   │   └── dataset_builder.py     # Train/val/test split builder
│   │
│   ├── features/
│   │   ├── code_embeddings.py     # CodeBERT-based embeddings
│   │   ├── ast_parser.py          # Tree-sitter AST extraction
│   │   └── syntax_tree_builder.py # Structured tree representations
│   │
│   ├── models/
│   │   ├── error_detector.py      # Multi-label error classifier
│   │   ├── explanation_model.py   # Fine-tuned explanation generator
│   │   ├── hint_generator.py      # Progressive hint model
│   │   └── model_factory.py       # Model loading and routing
│   │
│   ├── reasoning_engine/          # Core teaching intelligence
│   │   ├── error_classifier.py    # Error type → concept mapping
│   │   ├── explanation_generator.py
│   │   ├── hint_system.py         # Step-by-step hint builder
│   │   └── feedback_scorer.py     # Code quality scoring
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── trainer.py
│   │   ├── loss_functions.py
│   │   └── metrics.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── learning_effectiveness_score.py
│   │   └── error_analysis.py
│   │
│   ├── inference/
│   │   ├── pipeline.py            # ← Public API entrypoint
│   │   ├── predict.py
│   │   ├── explain.py
│   │   └── tutor_mode.py          # Interactive tutoring session
│   │
│   └── utils/
│       ├── logger.py
│       ├── config_loader.py
│       └── helpers.py
│
├── experiments/
│   ├── exp_01_tfidf_baseline/
│   ├── exp_02_codebert_model/
│   ├── exp_03_explanation_finetune/
│   └── logs/
│
├── models_saved/
│   ├── baseline_model.pkl
│   ├── codebert_model/
│   └── best_mentor_model.pt
│
├── tests/
│   ├── test_data_pipeline.py
│   ├── test_model.py
│   ├── test_explanations.py
│   └── test_inference.py
│
├── scripts/
│   ├── train.sh
│   ├── evaluate.sh
│   ├── preprocess.sh
│   └── run_tutor.sh
│
└── docs/
    ├── architecture.md
    ├── dataset_guide.md
    ├── explanation_system.md
    └── future_polycode_integration.md
```

---

## ⚡ Execution Flow

```
Step 1 — Data Pipeline
  raw code + error samples ──► collector.py ──► cleaner.py ──► dataset_builder.py

Step 2 — Feature Extraction
  tokenizer.py ──► ast_parser.py ──► code_embeddings.py ──► syntax_tree_builder.py

Step 3 — Model Training
  train.py ──► trainer.py ──► error_detector + explanation_model ──► models_saved/

Step 4 — Reasoning Engine
  error_classifier.py ──► explanation_generator.py ──► hint_system.py

Step 5 — Inference
  src/inference/pipeline.py  ◄── error + explanation + hint + score (fused output)
```

---

## 🔬 ML Approaches

### Baseline

- TF-IDF on code tokens + Logistic Regression
- Fast, interpretable, establishes performance floor
- Tracked in `exp_01_tfidf_baseline/`

### Primary Model (CodeBERT)

- Pre-trained on GitHub code across multiple languages
- Fine-tuned on labeled buggy/correct code pairs
- Multi-label classifier: a snippet can have multiple error types simultaneously
- Outputs confidence score per error type

### Explanation Generation

- Fine-tuned sequence-to-sequence model (CodeT5 / LLaMA-based)
- Input: code snippet + detected error type
- Output: plain-English explanation of what went wrong and why
- Trained on annotated error-explanation pairs from Stack Overflow and educational datasets

### Advanced (Planned)

- **Adaptive Difficulty**: User skill level estimated from error history; explanation depth adjusted dynamically
- **Code-to-Concept Graph**: Maps detected patterns to a concept graph (e.g., nested loop → time complexity → Big-O)
- **Feedback Learning Loop**: Model improves from user signals (hint accepted, explanation rated helpful, etc.)

---

## 🧪 Datasets

| Source                                                           | Description                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------------ |
| [CodeNet (IBM)](https://github.com/IBM/Project_CodeNet)          | 14M code samples across 55 languages with execution outcomes |
| [Stack Overflow Dump](https://archive.org/details/stackexchange) | Error questions + accepted answers for explanation training  |
| [LeetCode / HackerRank](https://github.com/your-org/polymentor)  | Problem + buggy solution pairs for hint system training      |
| [ManyBugs / IntroClass](https://repairbenchmarks.cs.umass.edu/)  | Labeled real-world bug datasets for error classification     |
| Custom Collected                                                 | Beginner code submissions with teacher-written explanations  |

All sources are normalized into `data/processed/train.json`, `val.json`, `test.json` via `src/data_pipeline/dataset_builder.py`.

---

## 🚀 Quickstart

### Prerequisites

- Python 3.10+
- GPU recommended for training (CPU works for inference)
- 8 GB RAM minimum · 16 GB recommended

### Installation

```bash
git clone https://github.com/your-org/polymentor.git
cd polymentor
pip install -r requirements.txt
pip install -e .
```

### Configuration

```bash
cp configs/model_config.yaml configs/model_config.local.yaml
# Edit as needed
```

### Run the Full Pipeline

```bash
# 1. Preprocess and build dataset
bash scripts/preprocess.sh

# 2. Train the error detection model
bash scripts/train.sh

# 3. Evaluate model performance
bash scripts/evaluate.sh

# 4. Launch interactive tutor mode
bash scripts/run_tutor.sh
```

### Python API

```python
from src.inference.pipeline import PolyMentorPipeline

mentor = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")

result = mentor.analyze("""
for i in range(10):
    print(i)
    if i = 5:
        break
""", language="python", level="beginner")

print(result.error_type)        # SyntaxError: assignment in condition
print(result.explanation)       # "You used = which assigns a value..."
print(result.hint)              # "Step 1: Think about what operator checks equality..."
print(result.concept_taught)    # "Comparison Operators: == vs ="
print(result.quality_score)     # 72 / 100
```

---

## 🧩 Tech Stack

| Layer                  | Technology                           |
| ---------------------- | ------------------------------------ |
| ML Framework           | PyTorch 2.x                          |
| Code Understanding     | CodeBERT (`microsoft/codebert-base`) |
| Explanation Generation | CodeT5 / LLaMA fine-tuned            |
| AST Parsing            | Tree-sitter (multi-language)         |
| Backend API            | FastAPI + Uvicorn                    |
| Deployment             | Docker + AWS                         |

---

## 📊 Error Types Detected

| Category              | Examples                                              |
| --------------------- | ----------------------------------------------------- |
| **Syntax Errors**     | Missing colons, unmatched brackets, wrong indentation |
| **Logical Errors**    | Off-by-one, wrong condition, infinite loop            |
| **Type Errors**       | Wrong data type passed, implicit conversion bugs      |
| **Runtime Patterns**  | Null reference, division by zero, out-of-bounds       |
| **Bad Practices**     | Deeply nested code, magic numbers, unused variables   |
| **Structural Issues** | Misused recursion, poor function decomposition        |

Error taxonomy is defined in `data/labels/error_types.json`.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Individual suites
pytest tests/test_data_pipeline.py
pytest tests/test_model.py
pytest tests/test_explanations.py
pytest tests/test_inference.py
```

---

## 📈 Experiments

| Experiment                    | Approach                       | Status         |
| ----------------------------- | ------------------------------ | -------------- |
| `exp_01_tfidf_baseline`       | TF-IDF + Logistic Regression   | ✅ Complete    |
| `exp_02_codebert_model`       | CodeBERT fine-tuned classifier | 🔄 In Progress |
| `exp_03_explanation_finetune` | Seq2Seq explanation generator  | 🗓️ Planned     |

---

## 🗺️ Roadmap

- [x] Project architecture and data pipeline design
- [x] Error taxonomy and difficulty label system
- [ ] CodeBERT fine-tuning on buggy code dataset
- [ ] Explanation generation model (CodeT5 / LLaMA)
- [ ] Progressive hint system with step-by-step output
- [ ] Adaptive difficulty scoring per user session
- [ ] FastAPI backend for real-time inference
- [ ] Polycode platform integration
- [ ] PolyGuard + PolyMentor unified intelligence layer
- [ ] Feedback learning loop (model improves from user signals)

---

## 🔗 Relationship to PolyGuard

| System            | Role                                                       |
| ----------------- | ---------------------------------------------------------- |
| 🛡️ **PolyGuard**  | Security · Vulnerability detection · Secure fix generation |
| 🧠 **PolyMentor** | Learning · Error explanation · Intelligent tutoring        |

Both systems share the same CodeBERT backbone and AST parsing infrastructure. Together they form the AI developer intelligence core of the **Polycode** platform.

---

## 📚 Documentation

- [`docs/architecture.md`](docs/architecture.md) — Full system design and component breakdown
- [`docs/dataset_guide.md`](docs/dataset_guide.md) — Dataset sources, collection, and labeling process
- [`docs/explanation_system.md`](docs/explanation_system.md) — How the explanation and hint models work
- [`docs/future_polycode_integration.md`](docs/future_polycode_integration.md) — Integration plan with Polycode platform

---

## 🤝 Contributing

Open an issue before submitting a PR. All contributions require tests and must pass the full suite.

```bash
pytest tests/ -v
flake8 src/ --max-line-length=100
```

---

## 📄 License

MIT License — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

Built with 🎓 by the QuantumLogics team.

_PolyMentor — Don't just fix code. Understand it._

</div>
