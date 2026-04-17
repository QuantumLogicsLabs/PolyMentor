# PolyMentor — Complete Beginner's Guide to the ML Model

> You don't need to know anything about AI or machine learning to understand this guide. We'll explain everything from scratch.

---

## Before We Start: What Even Is a "Model"?

When people say "the AI model", they mean a program that has been *trained* to recognize patterns.

Think of it like this: if you showed a child thousands of pictures of cats and dogs and told them which was which every time, eventually the child could look at a new picture and say "that's a cat" without you telling them. A machine learning model does the same thing — except instead of pictures of animals, PolyMentor looks at code, and instead of "cat or dog", it answers "does this code have an error, and if so, what kind?"

The "training" is just the process of showing the model many examples until it gets good at recognizing patterns on its own.

---

## Part 1: What Problem Are We Actually Solving?

Before we talk about the model, let's talk about the problem.

Imagine you are learning to write Python code and you write this:

```python
for i in range(10):
    if i = 5:
        break
```

A normal spell-checker would say: *"Error on line 2."*

That's not very helpful. PolyMentor is built to say something more like: *"You used `=` here, which sets a value. But inside an `if` statement, you probably want to check if something is equal — and for that, Python uses `==`. This is called the comparison operators concept."*

That's the whole goal. Not just finding errors, but explaining them the way a teacher would.

To do that, the system needs to:
1. Read code and understand its structure.
2. Decide what kind of error it is.
3. Generate a plain-English explanation.
4. Produce hints that help the learner think it through.

Each of these steps is handled by a different part of the model pipeline.

---

## Part 2: How the Model "Reads" Code

Before the model can do anything, it needs to convert code into a format a computer can process mathematically. Code is just text — and math doesn't work directly on letters and symbols. There are two steps to bridge that gap.

### Step 1: Building a Syntax Tree (AST Parsing)

An **AST** stands for **Abstract Syntax Tree**. Don't let the name intimidate you — it's just a way of representing the *structure* of code as a diagram.

Take this simple code:

```python
x = 5 + 3
```

A human reads this and understands: "assign to x the result of adding 5 and 3." A syntax tree represents that same understanding as a structured diagram:

```
Assignment
├── Variable: x
└── Addition
    ├── Number: 5
    └── Number: 3
```

PolyMentor uses a tool called **Tree-sitter** to build these trees automatically for C++, Python, JavaScript, and Java. Tree-sitter is fast and works even when the code has errors in it — which is important, because we often need to analyze broken code.

Why does this matter? Because many errors are invisible if you just look at the raw text, but become obvious when you look at the tree. A mismatched bracket, for example, shows up as a broken branch in the tree.

This is handled by `src/features/ast_parser.py` and `src/features/syntax_tree_builder.py`.

---

### Step 2: Turning Code Into Numbers (Embeddings)

Once we have the tree, we still need to convert everything into numbers, because the neural network at the core of the model only works with numbers.

This is done through something called **embeddings**. An embedding is a list of numbers (called a vector) that represents the *meaning* of a piece of text or code.

Here's an intuition for why this works: words that mean similar things end up with similar number patterns. The word "error" and the word "bug" will have vectors that are mathematically close to each other. The word "banana" will be far away from both.

PolyMentor uses a pre-trained model called **CodeBERT** (`microsoft/codebert-base`) to generate these embeddings. CodeBERT was trained by Microsoft on hundreds of millions of lines of real code from GitHub. It already "understands" a lot about code — PolyMentor builds on top of that understanding.

This is handled by `src/features/code_embeddings.py`.

---

## Part 3: The Error Detection Model

Now that code has been converted into a tree and then into numbers, the model can actually look for errors.

### What Kind of Model Is It?

PolyMentor uses a **multi-label classifier**. Let's break that down:

- A **classifier** is a model that sorts inputs into categories. An email spam filter is a classifier — it puts emails into "spam" or "not spam".
- A **multi-label** classifier can put one input into *multiple* categories at the same time. This is important because a single snippet of code can have more than one error simultaneously.

The classifier is built on top of CodeBERT. Instead of using CodeBERT's full output as-is, PolyMentor adds a small extra layer on top that produces one confidence score per error type. A score close to 1 means "very likely this error exists"; a score close to 0 means "probably not".

### What Does "Fine-Tuning" Mean?

CodeBERT was trained on code in general — it didn't specifically learn to detect errors. To make it good at *error detection*, PolyMentor takes the pre-trained CodeBERT and trains it further on a labeled dataset of buggy and correct code pairs.

This process is called **fine-tuning**. Think of it like hiring someone who already has a general software engineering degree and then giving them a few months of specialized training in code review. They already had the foundation — you just pointed their skills in a specific direction.

Fine-tuning is much faster and cheaper than training from scratch, because the model already understands a lot about code.

This is all handled by `src/models/error_detector.py`.

---

## Part 4: The Explanation Model

Detecting the error is only half the job. The other half is explaining it.

### What Is a Seq2Seq Model?

The explanation generator uses a type of model called a **sequence-to-sequence** (seq2seq) model. This is the same type of model used in machine translation — it takes a sequence of input (a sentence in French) and produces a sequence of output (the same sentence in English).

PolyMentor adapts this idea: the input sequence is the code snippet plus the detected error type, and the output sequence is a plain-English explanation.

The model used is either **CodeT5** or a fine-tuned **LLaMA** variant — both are capable seq2seq models that can handle code as input.

### How Was It Trained to Explain?

The explanation model was trained on pairs of:
- **Input:** buggy code + error type label
- **Output:** a human-written explanation of what went wrong and why

The training data came from Stack Overflow accepted answers and a custom dataset of teacher-written explanations. The model learned to produce explanations that sound like a teacher wrote them — because many of them *were* written by teachers.

This is handled by `src/models/explanation_model.py` and `src/reasoning_engine/explanation_generator.py`.

---

## Part 5: The Hint System

The hint system is designed around a simple idea: a good hint moves the learner forward without giving them the answer.

### How Are Hints Generated?

Hints are generated by a fine-tuned model that was trained on:
- A buggy code submission.
- The correct version of the same code.
- A sequence of hints ordered from most abstract to most revealing.

The hints were constructed by comparing the buggy and correct solutions and creating a series of questions that walk the learner from "I don't see the problem" to "I can see exactly what to change" — one step at a time.

The model learned from thousands of such examples and can now generate similar hint sequences for new code it has never seen before.

### Why Not Just Give the Answer?

Learning research consistently shows that people understand and retain information better when they figure things out themselves, even if they need prompting to get there. Giving the answer immediately short-circuits the learning process. The hint system is designed to preserve the "aha moment" while making sure the learner doesn't get stuck.

This is handled by `src/models/hint_generator.py` and `src/reasoning_engine/hint_system.py`.

---

## Part 6: The Concept Mapping

This is perhaps the most distinctive part of PolyMentor.

When an error is detected, the system doesn't just label it — it maps it to a *concept* the learner needs to understand. The goal is to answer: "What is the fundamental thing this person doesn't know yet?"

The mapping works like a family tree of concepts:

```
Specific Error: assignment_in_condition (used = instead of ==)
        │
        ▼
Concept: comparison_operators
        │
        ▼
Parent concept: control_flow_basics
        │
        ▼
Domain: fundamental_syntax
```

By surfacing this chain, the system can calibrate explanations: a beginner gets the full explanation from the bottom up; an advanced learner might only need the top-level concept name.

This is handled by `src/reasoning_engine/error_classifier.py`.

---

## Part 7: The Quality Score

Even when code runs correctly, it can still be written poorly. The quality scorer evaluates code on four dimensions:

| Dimension | What it checks |
|---|---|
| **Correctness** | Does the code have errors? |
| **Readability** | Are variable names clear? Is the indentation consistent? |
| **Complexity** | Is the code unnecessarily complicated? Too many nested loops? |
| **Clean code** | Are there magic numbers? Unused variables? Duplicated logic? |

Each dimension gets a sub-score. They are combined into a single 0–100 number. The scorer also generates specific, actionable suggestions — not just "improve readability", but "rename the variable `x` to something that describes what it stores".

This is handled by `src/reasoning_engine/feedback_scorer.py`.

---

## Part 8: How Training Works

Training is the process of making the model better by showing it examples. Here is what happens step by step.

### 1. Load the Data

The training script reads from `data/processed/train.json` — thousands of labeled examples of buggy code with their error types, explanations, and hints.

### 2. Forward Pass

The model looks at one example and makes a prediction. For example, it looks at a snippet of Python and guesses: "I think this is a `logical_error/off_by_one`."

### 3. Loss Calculation

The prediction is compared to the correct answer. The difference between the prediction and the truth is called the **loss**. A high loss means the model was very wrong. A loss near zero means it was very close.

This is handled by `src/training/loss_functions.py`.

### 4. Backward Pass (Backpropagation)

The model figures out which of its internal settings (called **weights**) caused the error. This is done through a mathematical process called backpropagation. You don't need to understand the math — the key idea is that the model nudges its weights in the direction that would have made the loss smaller.

### 5. Repeat

This process repeats for every example in the training set. One full pass through the entire dataset is called an **epoch**. Training typically runs for multiple epochs.

Over time, the model's weights settle into values that make good predictions across the whole dataset.

This is handled by `src/training/train.py` and `src/training/trainer.py`.

---

## Part 9: How We Know If It's Working (Evaluation)

Training alone doesn't tell you if the model is actually good. For that, we use the test set — a collection of examples the model has *never seen during training*.

The evaluation measures several things:

- **Accuracy** — What percentage of error types did the model get right?
- **F1 Score** — A combined measure of precision (when it says there's an error, is it right?) and recall (when there is an error, does it find it?).
- **Learning Effectiveness Score** — A custom metric that measures whether the explanations and hints actually helped learners fix their code in subsequent attempts.

The last metric is unique to PolyMentor. Most AI systems only measure whether the model was technically correct. PolyMentor also measures whether the output was *useful for learning* — which is the actual goal.

This is handled by `src/evaluation/evaluate.py` and `src/evaluation/learning_effectiveness_score.py`.

---

## Part 10: How to Run It

You don't need to understand everything above to use the model. Here is the minimum you need to do.

### Install

```bash
git clone https://github.com/your-org/polymentor.git
cd polymentor
pip install -r requirements.txt
pip install -e .
```

### Prepare Data

```bash
bash scripts/preprocess.sh
```

This downloads and processes the training datasets. It may take a while the first time.

### Train

```bash
bash scripts/train.sh
```

This will train the error detection model. A GPU will make this significantly faster, but it works on CPU too. Progress is logged to `experiments/logs/`.

### Evaluate

```bash
bash scripts/evaluate.sh
```

This runs the model against the held-out test set and prints the evaluation metrics.

### Use It

```bash
bash scripts/run_tutor.sh
```

This launches the interactive tutor. You paste in code, it analyzes it and responds.

Or in Python:

```python
from src.inference.pipeline import PolyMentorPipeline

mentor = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")

result = mentor.analyze("""
for i in range(10):
    if i = 5:
        break
""", language="python", level="beginner")

print(result.error_type)       # What went wrong
print(result.explanation)      # Why it went wrong
print(result.hint)             # First step toward fixing it
print(result.concept_taught)   # The concept to learn
print(result.quality_score)    # Overall code quality
```

---

## Glossary

Here are all the technical terms used in this guide, in plain English:

| Term | What it means |
|---|---|
| **Model** | A program trained on examples to recognize patterns and make predictions |
| **Training** | The process of showing a model many labeled examples so it learns to make good predictions |
| **Fine-tuning** | Taking a model that's already been trained on general data and training it further on specific data |
| **AST (Abstract Syntax Tree)** | A tree-shaped diagram representing the structure of code |
| **Embedding** | A list of numbers that represents the meaning of a word, sentence, or code snippet |
| **Classifier** | A model that sorts inputs into categories |
| **Multi-label classifier** | A classifier that can assign more than one category to a single input |
| **Seq2Seq (Sequence-to-Sequence)** | A model that takes a sequence of input and produces a sequence of output (like translation) |
| **CodeBERT** | A pre-trained model from Microsoft that understands code, used as the base for PolyMentor |
| **Loss** | A number that measures how wrong the model's prediction was — lower is better |
| **Backpropagation** | The mathematical process by which a model adjusts its internal settings after making a wrong prediction |
| **Epoch** | One full pass through the entire training dataset |
| **Weights** | The internal numbers inside a model that determine how it makes predictions — these are what training adjusts |
| **F1 Score** | A metric that balances how often the model is right when it predicts an error, and how often it catches all errors |
| **Tree-sitter** | A library that builds syntax trees from source code across many programming languages |
