"""
src/inference/predict.py
------------------------
Low-level forward pass wrappers.

This module sits between the pipeline and the raw model classes.
It handles everything that is mechanical but fiddly:
    - Tokenising raw code strings into model inputs
    - Moving tensors to the right device
    - Batching
    - Running the forward pass
    - Decoding model outputs back to Python objects

The pipeline calls predict.py; predict.py calls the models.
Nothing outside src/inference/ should need to call this module directly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from src.models.error_detector import ErrorDetectionOutput, ErrorDetector, ErrorLabelRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# CodeBERT max token length — hard limit from BERT architecture
DETECTOR_MAX_LENGTH = 512

# Explanation / hint generation limits
EXPLANATION_MAX_INPUT_LENGTH  = 512
EXPLANATION_MAX_OUTPUT_TOKENS = 256

HINT_MAX_INPUT_LENGTH  = 512
HINT_MAX_OUTPUT_TOKENS = 128

# Number of hint steps to generate in one call
DEFAULT_NUM_HINTS = 3

# Beam search width — larger = better quality, slower
GENERATION_NUM_BEAMS = 4


# ---------------------------------------------------------------------------
# Prediction result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class DetectionResult:
    """
    Output of predict_errors() for a single code snippet.

    Attributes
    ----------
    error_labels:
        List of predicted error type strings, e.g.
        ["syntax_error/missing_colon", "logical_error/off_by_one"].
        Empty list means no errors were detected.
    confidences:
        Dictionary mapping each predicted label to its confidence score (0–1).
    has_error:
        True if at least one error was detected above the threshold.
    raw_probabilities:
        Full probability vector over all labels, shape (num_labels,).
        Useful for analysis or secondary thresholding.
    """
    error_labels:       List[str]
    confidences:        dict[str, float]
    has_error:          bool
    raw_probabilities:  torch.Tensor


@dataclass
class ExplanationResult:
    """
    Output of predict_explanation() for a single error.

    Attributes
    ----------
    explanation:
        Plain-English explanation of why the error occurred.
    input_tokens:
        Number of tokens in the model input (useful for debugging truncation).
    """
    explanation:   str
    input_tokens:  int


@dataclass
class HintResult:
    """
    Output of predict_hints() for a single error.

    Attributes
    ----------
    hints:
        Ordered list of hint strings, from most abstract to most revealing.
        The first hint is shown by default; subsequent ones are revealed on demand.
    total_hints:
        Total number of hints generated.
    """
    hints:        List[str]
    total_hints:  int


# ---------------------------------------------------------------------------
# Input formatting helpers
# ---------------------------------------------------------------------------

def _format_detector_input(code: str, language: str) -> str:
    """
    Prepend a language tag to the code string.

    CodeBERT was pre-trained on raw code. Adding a language tag as a prefix
    gives the model a strong signal about the syntax it should expect.

    Example output:
        "<python> for i in range(10):\n    if i = 5:\n        break"
    """
    return f"<{language}> {code.strip()}"


def _format_explanation_input(
    code: str,
    language: str,
    error_label: str,
    concept: str,
    level: str,
) -> str:
    """
    Build the input string for the explanation model.

    The model was fine-tuned on inputs with this exact format, so changing
    the structure here will degrade explanation quality.

    Example output:
        "[LANG] python [LEVEL] beginner [ERROR] syntax_error/missing_colon
         [CONCEPT] comparison_operators [CODE] if i = 5: break"
    """
    return (
        f"[LANG] {language} "
        f"[LEVEL] {level} "
        f"[ERROR] {error_label} "
        f"[CONCEPT] {concept} "
        f"[CODE] {code.strip()}"
    )


def _format_hint_input(
    code: str,
    language: str,
    error_label: str,
    concept: str,
    level: str,
) -> str:
    """
    Build the input string for the hint generator.

    Similar structure to the explanation input but with a [TASK] tag
    that tells the model to produce hints rather than explanations.
    """
    return (
        f"[TASK] hint "
        f"[LANG] {language} "
        f"[LEVEL] {level} "
        f"[ERROR] {error_label} "
        f"[CONCEPT] {concept} "
        f"[CODE] {code.strip()}"
    )


# ---------------------------------------------------------------------------
# Core prediction functions
# ---------------------------------------------------------------------------

def predict_errors(
    code: str,
    language: str,
    model: ErrorDetector,
    tokenizer: PreTrainedTokenizer,
    registry: ErrorLabelRegistry,
    device: torch.device,
    threshold: float = 0.5,
) -> DetectionResult:
    """
    Run the error detection model on a single code snippet.

    This function handles tokenisation, tensor construction, the forward
    pass, and decoding — all in one call.

    Args:
        code:       Raw source code string.
        language:   Programming language tag (python, javascript, cpp, java).
        model:      Loaded ErrorDetector instance (already on device, eval mode).
        tokenizer:  CodeBERT tokenizer.
        registry:   ErrorLabelRegistry for decoding label indices to strings.
        device:     The torch device the model is on.
        threshold:  Confidence threshold. Predictions above this are positive.

    Returns:
        DetectionResult with predicted error types and confidence scores.
    """
    # Format and tokenise
    text = _format_detector_input(code, language)

    encoding = tokenizer(
        text,
        max_length=DETECTOR_MAX_LENGTH,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Forward pass — no gradient needed for inference
    with torch.no_grad():
        output: ErrorDetectionOutput = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            threshold=threshold,
            registry=registry,
        )

    # output.predicted_labels is a list of lists (batch dim = 1)
    error_labels = output.predicted_labels[0] if output.predicted_labels else []
    probs_vector = output.probabilities[0]  # shape (num_labels,)

    # Build label → confidence mapping for detected labels only
    confidences: dict[str, float] = {
        label: probs_vector[registry.label_to_idx(label)].item()
        for label in error_labels
    }

    return DetectionResult(
        error_labels=error_labels,
        confidences=confidences,
        has_error=len(error_labels) > 0,
        raw_probabilities=probs_vector.cpu(),
    )


def predict_explanation(
    code: str,
    language: str,
    error_label: str,
    concept: str,
    level: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
) -> ExplanationResult:
    """
    Generate a plain-English explanation for a detected error.

    Args:
        code:         The buggy code snippet.
        language:     Programming language.
        error_label:  The primary error type string (e.g. "syntax_error/missing_colon").
        concept:      The concept this error maps to (from the reasoning engine).
        level:        Learner level — "beginner", "intermediate", or "advanced".
        model:        Loaded explanation model (seq2seq, on device, eval mode).
        tokenizer:    Explanation model tokenizer.
        device:       Torch device.

    Returns:
        ExplanationResult with the generated explanation text.
    """
    prompt = _format_explanation_input(code, language, error_label, concept, level)

    encoding = tokenizer(
        prompt,
        max_length=EXPLANATION_MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    input_token_count = input_ids.shape[1]

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=EXPLANATION_MAX_OUTPUT_TOKENS,
            num_beams=GENERATION_NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    explanation = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()

    if not explanation:
        # Fallback template — used before fine-tuning is complete
        explanation = _fallback_explanation(error_label, language)
        logger.debug("Explanation model produced empty output; using fallback template.")

    return ExplanationResult(
        explanation=explanation,
        input_tokens=input_token_count,
    )


def predict_hints(
    code: str,
    language: str,
    error_label: str,
    concept: str,
    level: str,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    num_hints: int = DEFAULT_NUM_HINTS,
) -> HintResult:
    """
    Generate a sequence of progressive hints for a detected error.

    Hints are returned as an ordered list. The first hint is the most
    abstract; each subsequent hint reveals more information.

    Args:
        code:         The buggy code snippet.
        language:     Programming language.
        error_label:  Primary error type string.
        concept:      Concept the error maps to.
        level:        Learner level.
        model:        Loaded hint generator (seq2seq, on device, eval mode).
        tokenizer:    Hint model tokenizer.
        device:       Torch device.
        num_hints:    How many hints to generate.

    Returns:
        HintResult with an ordered list of hint strings.
    """
    prompt = _format_hint_input(code, language, error_label, concept, level)

    encoding = tokenizer(
        prompt,
        max_length=HINT_MAX_INPUT_LENGTH,
        truncation=True,
        return_tensors="pt",
    )

    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        # Generate num_hints separate sequences via beam search with num_return_sequences
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=HINT_MAX_OUTPUT_TOKENS,
            num_beams=max(GENERATION_NUM_BEAMS, num_hints),
            num_return_sequences=num_hints,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    hints: List[str] = []
    for seq in generated_ids:
        decoded = tokenizer.decode(
            seq,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        if decoded:
            hints.append(decoded)

    if not hints:
        hints = _fallback_hints(error_label, level)
        logger.debug("Hint model produced empty output; using fallback hints.")

    return HintResult(hints=hints, total_hints=len(hints))


def predict_batch_errors(
    code_list: List[str],
    language: str,
    model: ErrorDetector,
    tokenizer: PreTrainedTokenizer,
    registry: ErrorLabelRegistry,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 8,
) -> List[DetectionResult]:
    """
    Run error detection on a list of code snippets efficiently.

    Processes snippets in batches to maximise GPU utilisation.
    Returns results in the same order as the input list.

    Args:
        code_list:  List of raw code strings.
        language:   All snippets must be in the same language.
                    For mixed-language batches, call predict_errors() per snippet.
        batch_size: Number of snippets per GPU batch.

    Returns:
        List of DetectionResult, one per input snippet.
    """
    results: List[DetectionResult] = []

    for start in range(0, len(code_list), batch_size):
        batch_codes = code_list[start : start + batch_size]
        texts = [_format_detector_input(c, language) for c in batch_codes]

        encoding = tokenizer(
            texts,
            max_length=DETECTOR_MAX_LENGTH,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        input_ids      = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            output: ErrorDetectionOutput = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                threshold=threshold,
                registry=registry,
            )

        for i in range(len(batch_codes)):
            error_labels  = output.predicted_labels[i] if output.predicted_labels else []
            probs_vector  = output.probabilities[i]
            confidences   = {
                label: probs_vector[registry.label_to_idx(label)].item()
                for label in error_labels
            }
            results.append(
                DetectionResult(
                    error_labels=error_labels,
                    confidences=confidences,
                    has_error=len(error_labels) > 0,
                    raw_probabilities=probs_vector.cpu(),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Fallback templates
# ---------------------------------------------------------------------------
# These are used when the fine-tuned models haven't been trained yet or
# produce empty output. They are intentionally generic — they tell the
# learner something is wrong and point them in the right direction.

def _fallback_explanation(error_label: str, language: str) -> str:
    category = error_label.split("/")[0] if "/" in error_label else error_label
    return (
        f"It looks like your {language} code has a {category.replace('_', ' ')}. "
        f"Review the flagged line carefully and compare it to the correct syntax "
        f"for {language}. The explanation model is still being trained — "
        f"more detailed guidance will be available soon."
    )


def _fallback_hints(error_label: str, level: str) -> List[str]:
    category = error_label.split("/")[0] if "/" in error_label else error_label
    return [
        f"Look closely at the flagged line. Does the {category.replace('_', ' ')} "
        f"match what you intended to do?",
        "Try reading the line out loud as a sentence. Does it say what you mean?",
        "Check the language documentation for the correct syntax for this pattern.",
    ]
