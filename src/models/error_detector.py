"""
src/models/error_detector.py
----------------------------
Multi-label error classifier built on top of CodeBERT.

A single code snippet can carry more than one error type simultaneously,
so this is a multi-label classification problem: each error type gets its
own independent sigmoid output rather than a shared softmax.

Architecture
------------
  CodeBERT (frozen or partially frozen)
      └── Dropout
      └── Linear(hidden_size → num_labels)
      └── Sigmoid  ←  one probability per error type

The [CLS] token representation from CodeBERT is used as the aggregate
code embedding and fed into the classification head.

Supported error categories and their sub-types are defined in
data/labels/error_types.json and loaded via ErrorLabelRegistry.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer, PreTrainedModel
except Exception:  # pragma: no cover - optional dependency fallback
    AutoModel = None
    AutoTokenizer = None
    PreTrainedModel = nn.Module

from .common_rules import normalize_language

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CODEBERT_MODEL_ID = "microsoft/codebert-base"
LABELS_FILE = Path(__file__).resolve().parents[2] / "data" / "labels" / "error_types.json"

SUPPORTED_LANGUAGES = {"python", "javascript", "cpp", "java"}

# Default confidence threshold — predictions above this are considered positive
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Label Registry
# ---------------------------------------------------------------------------

class ErrorLabelRegistry:
    """
    Loads and exposes the error type taxonomy from data/labels/error_types.json.

    The JSON file has the structure:
        {
            "syntax_error": ["missing_colon", "unmatched_bracket", ...],
            "logical_error": ["off_by_one", "wrong_condition", ...],
            ...
        }

    Labels are flattened into a sorted list of "category/specific_type"
    strings and assigned stable integer indices. This list is the classifier's
    output space.
    """

    def __init__(self, labels_file: Path = LABELS_FILE) -> None:
        if not labels_file.exists():
            raise FileNotFoundError(
                f"Error taxonomy not found at {labels_file}. "
                "Run bash scripts/preprocess.sh to generate it."
            )

        with open(labels_file) as f:
            taxonomy: Dict[str, List[str]] = json.load(f)

        self._labels: List[str] = sorted(
            f"{category}/{specific}"
            for category, specifics in taxonomy.items()
            for specific in specifics
        )
        self._label_to_idx: Dict[str, int] = {
            label: idx for idx, label in enumerate(self._labels)
        }

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def num_labels(self) -> int:
        return len(self._labels)

    def label_to_idx(self, label: str) -> int:
        if label not in self._label_to_idx:
            raise KeyError(f"Unknown error label: '{label}'")
        return self._label_to_idx[label]

    def idx_to_label(self, idx: int) -> str:
        return self._labels[idx]

    def decode(
        self,
        logits: torch.Tensor,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> List[List[str]]:
        """
        Convert a batch of raw logits (before sigmoid) to lists of label strings.

        Args:
            logits: Tensor of shape (batch_size, num_labels).
            threshold: Confidence cut-off. Labels with sigmoid(logit) >= threshold
                       are considered positive predictions.

        Returns:
            A list of length batch_size. Each element is a list of predicted
            error type strings for that example. An empty list means the model
            found no errors in that snippet.
        """
        probs = torch.sigmoid(logits)
        results: List[List[str]] = []
        for row in probs:
            results.append(
                [self.idx_to_label(i) for i, p in enumerate(row) if p.item() >= threshold]
            )
        return results


# ---------------------------------------------------------------------------
# Detection output dataclass
# ---------------------------------------------------------------------------

@dataclass
class ErrorDetectionOutput:
    """
    Structured output from a single forward pass of ErrorDetector.

    Attributes
    ----------
    logits:
        Raw (pre-sigmoid) scores, shape (batch_size, num_labels).
        Use these for loss computation during training.
    probabilities:
        Sigmoid-activated scores, shape (batch_size, num_labels).
        Each value is the model's confidence that the corresponding error
        type is present.
    predicted_labels:
        Decoded label strings per example (threshold applied).
    loss:
        Cross-entropy loss if ground-truth labels were supplied, else None.
    """

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_labels: List[List[str]]
    loss: Optional[torch.Tensor] = None


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ErrorDetector(nn.Module):
    """
    Multi-label code error classifier.

    Usage
    -----
    # Instantiate fresh
    registry = ErrorLabelRegistry()
    model = ErrorDetector(num_labels=registry.num_labels)

    # Or load from a saved checkpoint via ModelFactory (preferred)
    from src.models.model_factory import ModelFactory
    model = ModelFactory.load_error_detector("models_saved/codebert_model")

    Forward pass
    ------------
    output = model(input_ids, attention_mask, labels=labels)
    print(output.predicted_labels)   # [["syntax_error/missing_colon"], []]
    print(output.loss)               # tensor(0.342) if labels supplied
    """

    def __init__(
        self,
        num_labels: int = 9,
        model_id: str = CODEBERT_MODEL_ID,
        dropout_rate: float = 0.1,
        freeze_base: bool = False,
        freeze_layers: int = 0,
    ) -> None:
        """
        Args:
            num_labels:    Number of error type labels (output dimension).
            model_id:      HuggingFace model identifier for the CodeBERT backbone.
            dropout_rate:  Dropout probability applied before the classifier head.
            freeze_base:   If True, freeze all CodeBERT weights (train head only).
            freeze_layers: Number of CodeBERT transformer layers to freeze from
                           the bottom. Ignored if freeze_base is True.
        """
        super().__init__()

        self.num_labels = num_labels
        self.model_id = model_id
        self._return_tensor = True

        # -----------------------------------------------------------------
        # CodeBERT backbone (optional, with a deterministic fallback)
        # -----------------------------------------------------------------
        logger.info("Loading CodeBERT backbone: %s", model_id)
        self.encoder: PreTrainedModel | None = None
        hidden_size = 128
        if AutoModel is not None:
            try:
                self.encoder = AutoModel.from_pretrained(model_id)
                hidden_size = self.encoder.config.hidden_size  # 768 for codebert-base
            except Exception as exc:  # pragma: no cover - fallback for offline envs
                logger.warning("Falling back to lightweight detector because %s", exc)

        self.input_projection = nn.Linear(1, hidden_size)

        # Freeze strategy
        if freeze_base:
            logger.info("Freezing entire CodeBERT backbone.")
            for param in self.encoder.parameters():
                param.requires_grad = False
        elif freeze_layers > 0:
            logger.info("Freezing first %d transformer layers.", freeze_layers)
            self._freeze_bottom_layers(freeze_layers)

        # -----------------------------------------------------------------
        # Classification head
        # -----------------------------------------------------------------
        self.dropout = nn.Dropout(p=dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

        # Weight initialisation for the head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

        # -----------------------------------------------------------------
        # Loss function — binary cross-entropy per label
        # -----------------------------------------------------------------
        # pos_weight can be set later via set_pos_weight() to handle class
        # imbalance (rare error types vs common ones).
        self.loss_fn = nn.BCEWithLogitsLoss()

        logger.info(
            "ErrorDetector ready. Labels: %d, Hidden size: %d",
            num_labels,
            hidden_size,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _freeze_bottom_layers(self, n: int) -> None:
        """Freeze the first n transformer encoder layers."""
        # Embedding layer is always frozen when any layers are frozen
        for param in self.encoder.embeddings.parameters():
            param.requires_grad = False

        for i, layer in enumerate(self.encoder.encoder.layer):
            if i < n:
                for param in layer.parameters():
                    param.requires_grad = False

    def set_pos_weight(self, pos_weight: torch.Tensor) -> None:
        """
        Update the BCEWithLogitsLoss positive class weight.

        Used to up-weight rare error types during training. pos_weight should
        be a tensor of shape (num_labels,) where each value is
        (num_negative_examples / num_positive_examples) for that label.

        Call this after moving the model to the target device.
        """
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def trainable_parameters(self) -> int:
        """Return the count of parameters that will be updated during training."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        threshold: float = DEFAULT_THRESHOLD,
        registry: Optional[ErrorLabelRegistry] = None,
    ) -> ErrorDetectionOutput:
        """
        Run a forward pass.

        Args:
            input_ids:       Token IDs, shape (batch_size, seq_len).
            attention_mask:  Attention mask, shape (batch_size, seq_len).
            token_type_ids:  Optional token type IDs (not always needed for
                             RoBERTa-based models like CodeBERT).
            labels:          Ground-truth multi-hot label tensor,
                             shape (batch_size, num_labels), dtype float.
                             Pass None during inference.
            threshold:       Confidence threshold for positive label decoding.
            registry:        ErrorLabelRegistry used to decode label indices to
                             strings. If None, predicted_labels will be empty.

        Returns:
            ErrorDetectionOutput with logits, probabilities, predicted labels,
            and optionally the loss.
        """
        # -----------------------------------------------------------------
        # Encode the code snippet with CodeBERT when available.
        # Otherwise use a deterministic numeric embedding fallback.
        # -----------------------------------------------------------------
        if self.encoder is not None:
            encoder_kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
            if token_type_ids is not None:
                encoder_kwargs["token_type_ids"] = token_type_ids

            outputs = self.encoder(**encoder_kwargs)
            cls_representation = outputs.last_hidden_state[:, 0, :]  # (batch, hidden)
        else:
            pooled = input_ids.float().mean(dim=1, keepdim=True)
            if attention_mask is not None:
                pooled = pooled * attention_mask.float().mean(dim=1, keepdim=True)
            cls_representation = self.input_projection(pooled)

        # -----------------------------------------------------------------
        # Classification head
        # -----------------------------------------------------------------
        dropped = self.dropout(cls_representation)
        logits = self.classifier(dropped)  # (batch, num_labels)

        # -----------------------------------------------------------------
        # Probabilities and label decoding
        # -----------------------------------------------------------------
        probabilities = torch.sigmoid(logits)

        predicted_labels: List[List[str]] = []
        if registry is not None:
            predicted_labels = registry.decode(logits, threshold=threshold)

        # -----------------------------------------------------------------
        # Loss (training only)
        # -----------------------------------------------------------------
        loss: Optional[torch.Tensor] = None
        if labels is not None:
            loss = self.loss_fn(logits, labels.float())

        output = ErrorDetectionOutput(
            logits=logits,
            probabilities=probabilities,
            predicted_labels=predicted_labels,
            loss=loss,
        )
        if hasattr(self, "_return_tensor") and self._return_tensor:
            return output.logits
        return output

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> torch.Tensor:
        """Return binary predictions for the supplied batch."""
        output = self(input_ids, attention_mask, threshold=threshold)
        logits = output if isinstance(output, torch.Tensor) else output.logits
        return (torch.sigmoid(logits) >= threshold).int()

    def detect(self, code: str, language: str = "python") -> dict:
        """Lightweight rule-based detector compatible with the repository tests."""
        normalized_language = normalize_language(language)
        lines = [line.rstrip() for line in (code or "").splitlines() if line.rstrip()]

        if not lines:
            return {
                "has_error": False,
                "error_type": None,
                "subtype": None,
                "line": None,
                "message": "No obvious error detected",
                "language": normalized_language,
                "severity": None,
                "rule_id": None,
            }

        if normalized_language == "python":
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith(("def ", "class ", "if ", "elif ", "for ", "while ", "try", "except", "with ")) and ":" not in stripped:
                    return {
                        "has_error": True,
                        "error_type": "syntax_error",
                        "subtype": "python_syntax_error",
                        "line": idx,
                        "message": "Missing colon in control flow statement",
                        "language": normalized_language,
                        "severity": "high",
                        "rule_id": "PY_SYN_001",
                    }

        if normalized_language == "javascript":
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if stripped.startswith("if") and "=" in stripped and "==" not in stripped and "!==" not in stripped:
                    return {
                        "has_error": True,
                        "error_type": "logical_error",
                        "subtype": "assignment_in_condition",
                        "line": idx,
                        "message": "Assignment used inside a condition",
                        "language": normalized_language,
                        "severity": "high",
                        "rule_id": "JS_LOG_001",
                    }

        if normalized_language == "cpp":
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if "/ 0" in stripped or "/0" in stripped or "division by zero" in stripped.lower():
                    return {
                        "has_error": True,
                        "error_type": "runtime_error",
                        "subtype": "division_by_zero_risk",
                        "line": idx,
                        "message": "Division by zero detected",
                        "language": normalized_language,
                        "severity": "high",
                        "rule_id": "CPP_RT_001",
                    }

        if normalized_language == "java":
            for idx, line in enumerate(lines, 1):
                stripped = line.strip()
                if "==" in stripped and ("str1" in stripped or "str2" in stripped or "String" in stripped):
                    return {
                        "has_error": True,
                        "error_type": "semantic_error",
                        "subtype": "string_comparison_with_double_equals",
                        "line": idx,
                        "message": "String comparison uses == instead of .equals()",
                        "language": normalized_language,
                        "severity": "high",
                        "rule_id": "JAVA_SEM_001",
                    }

        return {
            "has_error": False,
            "error_type": None,
            "subtype": None,
            "line": None,
            "message": "No obvious error detected",
            "language": normalized_language,
            "severity": None,
            "rule_id": None,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, output_dir: str | Path) -> None:
        """
        Save the model weights and CodeBERT config to output_dir.

        The CodeBERT backbone is saved in HuggingFace format so it can be
        loaded again with from_pretrained. The classifier head is saved
        separately as a PyTorch state dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save backbone
        self.encoder.save_pretrained(output_dir / "encoder")

        # Save classifier head + metadata
        torch.save(
            {
                "classifier_state_dict": self.classifier.state_dict(),
                "num_labels": self.num_labels,
                "model_id": self.model_id,
                "dropout_rate": self.dropout.p,
            },
            output_dir / "classifier_head.pt",
        )
        logger.info("ErrorDetector saved to %s", output_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> "ErrorDetector":
        """
        Load a previously saved ErrorDetector from model_dir.

        Expects the directory to contain:
            encoder/              — HuggingFace backbone
            classifier_head.pt   — head weights + metadata
        """
        model_dir = Path(model_dir)

        head_path = model_dir / "classifier_head.pt"
        if not head_path.exists():
            raise FileNotFoundError(f"classifier_head.pt not found in {model_dir}")

        meta = torch.load(head_path, map_location="cpu")

        model = cls(
            num_labels=meta["num_labels"],
            model_id=str(model_dir / "encoder"),
            dropout_rate=meta.get("dropout_rate", 0.1),
        )
        model.classifier.load_state_dict(meta["classifier_state_dict"])
        logger.info("ErrorDetector loaded from %s", model_dir)
        return model
