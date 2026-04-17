"""
src/models/model_factory.py
---------------------------
Central model loader and router.

All model loading goes through ModelFactory. No other module in the codebase
should import model classes and call their constructors directly — they should
go through the factory. This keeps device placement, checkpoint resolution,
and version management in one place.

Supported model types
---------------------
    error_detector   — CodeBERT multi-label error classifier
    explanation      — CodeT5 / LLaMA seq2seq explanation generator
    hint             — Progressive hint generator

Usage
-----
    from src.models.model_factory import ModelFactory

    # Load everything for inference
    factory = ModelFactory.from_checkpoint("models_saved/best_mentor_model.pt")
    detector    = factory.error_detector
    explainer   = factory.explanation_model
    hint_gen    = factory.hint_generator

    # Or load a single model
    detector = ModelFactory.load_error_detector("models_saved/codebert_model")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.models.error_detector import ErrorDetector, ErrorLabelRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

ModelType = Literal["error_detector", "explanation", "hint"]

# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT     = _ROOT / "models_saved" / "best_mentor_model.pt"
DEFAULT_DETECTOR_DIR   = _ROOT / "models_saved" / "codebert_model"
DEFAULT_EXPLANATION_DIR = _ROOT / "models_saved" / "explanation_model"
DEFAULT_LABELS_FILE    = _ROOT / "data" / "labels" / "error_types.json"

# Explanation backbone — CodeT5 small is the default; swap for LLaMA if configured
DEFAULT_EXPLANATION_MODEL_ID = "Salesforce/codet5-base"
DEFAULT_HINT_MODEL_ID        = "Salesforce/codet5-base"


# ---------------------------------------------------------------------------
# Loaded model bundle
# ---------------------------------------------------------------------------

@dataclass
class ModelBundle:
    """
    Container returned by ModelFactory.from_checkpoint().

    Holds references to all three models and their associated tokenizers,
    plus the label registry. All models are on the same device.
    """
    error_detector:        ErrorDetector
    detector_tokenizer:    PreTrainedTokenizer
    explanation_model:     PreTrainedModel
    explanation_tokenizer: PreTrainedTokenizer
    hint_generator:        PreTrainedModel
    hint_tokenizer:        PreTrainedTokenizer
    label_registry:        ErrorLabelRegistry
    device:                torch.device


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """
    Central loader and router for all PolyMentor models.

    Class methods
    -------------
    from_checkpoint(path)       — Load the fused best_mentor_model.pt bundle.
    load_error_detector(dir)    — Load only the error detection model.
    load_explanation_model(dir) — Load only the explanation model.
    load_hint_generator(dir)    — Load only the hint generator.
    resolve_device()            — Detect and return the best available device.

    All load methods accept either a directory path (for HuggingFace-style
    checkpoints) or a .pt file path (for PyTorch state dict checkpoints).
    """

    # ------------------------------------------------------------------
    # Device resolution
    # ------------------------------------------------------------------

    @staticmethod
    def resolve_device(prefer_gpu: bool = True) -> torch.device:
        """
        Return the best available torch device.

        Priority: CUDA → MPS (Apple Silicon) → CPU.
        Set prefer_gpu=False to force CPU (useful for debugging).
        """
        if not prefer_gpu:
            logger.info("Device forced to CPU.")
            return torch.device("cpu")

        if torch.cuda.is_available():
            device = torch.device("cuda")
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("CUDA device found: %s (%.1f GB VRAM)", gpu_name, vram_gb)
            return device

        if torch.backends.mps.is_available():
            logger.info("Apple MPS device found.")
            return torch.device("mps")

        logger.warning(
            "No GPU found. Running on CPU. "
            "Training will be slow; inference is fine for small snippets."
        )
        return torch.device("cpu")

    # ------------------------------------------------------------------
    # Fused checkpoint
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        device: Optional[torch.device] = None,
        prefer_gpu: bool = True,
    ) -> ModelBundle:
        """
        Load the fused best_mentor_model.pt checkpoint.

        This is a PyTorch file saved by src/training/trainer.py --finalize.
        It contains the state dicts of all three models plus metadata.

        Args:
            checkpoint_path: Path to best_mentor_model.pt.
            device:          Target device. If None, auto-detected.
            prefer_gpu:      If True, use GPU when available.

        Returns:
            ModelBundle with all models ready for inference.
        """
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}\n"
                "Run bash scripts/train.sh to create it."
            )

        if device is None:
            device = cls.resolve_device(prefer_gpu=prefer_gpu)

        logger.info("Loading fused checkpoint from %s ...", checkpoint_path)
        bundle_data = torch.load(checkpoint_path, map_location=device)

        # -----------------------------------------------------------------
        # Error detector
        # -----------------------------------------------------------------
        registry = ErrorLabelRegistry()

        detector = ErrorDetector(
            num_labels=bundle_data["detector_num_labels"],
            model_id=bundle_data.get("detector_model_id", "microsoft/codebert-base"),
        )
        detector.load_state_dict(bundle_data["detector_state_dict"])
        detector.to(device)
        detector.eval()

        detector_tokenizer = AutoTokenizer.from_pretrained(
            bundle_data.get("detector_tokenizer_id", "microsoft/codebert-base")
        )

        # -----------------------------------------------------------------
        # Explanation model
        # -----------------------------------------------------------------
        explanation_model_id = bundle_data.get(
            "explanation_model_id", DEFAULT_EXPLANATION_MODEL_ID
        )
        explanation_model = AutoModelForSeq2SeqLM.from_pretrained(explanation_model_id)
        if "explanation_state_dict" in bundle_data:
            explanation_model.load_state_dict(bundle_data["explanation_state_dict"])
        explanation_model.to(device)
        explanation_model.eval()

        explanation_tokenizer = AutoTokenizer.from_pretrained(explanation_model_id)

        # -----------------------------------------------------------------
        # Hint generator
        # -----------------------------------------------------------------
        hint_model_id = bundle_data.get("hint_model_id", DEFAULT_HINT_MODEL_ID)
        hint_generator = AutoModelForSeq2SeqLM.from_pretrained(hint_model_id)
        if "hint_state_dict" in bundle_data:
            hint_generator.load_state_dict(bundle_data["hint_state_dict"])
        hint_generator.to(device)
        hint_generator.eval()

        hint_tokenizer = AutoTokenizer.from_pretrained(hint_model_id)

        logger.info("All models loaded and ready on %s.", device)

        return ModelBundle(
            error_detector=detector,
            detector_tokenizer=detector_tokenizer,
            explanation_model=explanation_model,
            explanation_tokenizer=explanation_tokenizer,
            hint_generator=hint_generator,
            hint_tokenizer=hint_tokenizer,
            label_registry=registry,
            device=device,
        )

    # ------------------------------------------------------------------
    # Individual model loaders
    # ------------------------------------------------------------------

    @classmethod
    def load_error_detector(
        cls,
        model_dir: str | Path = DEFAULT_DETECTOR_DIR,
        device: Optional[torch.device] = None,
        prefer_gpu: bool = True,
    ) -> tuple[ErrorDetector, PreTrainedTokenizer, ErrorLabelRegistry]:
        """
        Load only the error detection model.

        Returns:
            (ErrorDetector, tokenizer, ErrorLabelRegistry)
        """
        model_dir = Path(model_dir)
        if not model_dir.exists():
            raise FileNotFoundError(
                f"Error detector directory not found: {model_dir}\n"
                "Run bash scripts/train.sh to train it."
            )

        if device is None:
            device = cls.resolve_device(prefer_gpu=prefer_gpu)

        logger.info("Loading ErrorDetector from %s ...", model_dir)
        model = ErrorDetector.load(model_dir)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir / "encoder"),
            local_files_only=True,
        )
        registry = ErrorLabelRegistry()

        logger.info("ErrorDetector loaded on %s.", device)
        return model, tokenizer, registry

    @classmethod
    def load_explanation_model(
        cls,
        model_dir: str | Path = DEFAULT_EXPLANATION_DIR,
        model_id: str = DEFAULT_EXPLANATION_MODEL_ID,
        device: Optional[torch.device] = None,
        prefer_gpu: bool = True,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load only the explanation generation model.

        If model_dir exists and contains a fine-tuned checkpoint, it is loaded.
        Otherwise falls back to the base pre-trained model_id. This allows the
        system to run with a template-based fallback before fine-tuning is done.

        Returns:
            (explanation_model, tokenizer)
        """
        if device is None:
            device = cls.resolve_device(prefer_gpu=prefer_gpu)

        model_dir = Path(model_dir)
        source = str(model_dir) if model_dir.exists() else model_id

        if not model_dir.exists():
            logger.warning(
                "Fine-tuned explanation model not found at %s. "
                "Loading base model %s. Explanations will be generic.",
                model_dir,
                model_id,
            )

        logger.info("Loading explanation model from %s ...", source)
        model = AutoModelForSeq2SeqLM.from_pretrained(source)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(source)

        logger.info("Explanation model loaded on %s.", device)
        return model, tokenizer

    @classmethod
    def load_hint_generator(
        cls,
        model_dir: Optional[str | Path] = None,
        model_id: str = DEFAULT_HINT_MODEL_ID,
        device: Optional[torch.device] = None,
        prefer_gpu: bool = True,
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        """
        Load only the hint generator model.

        Returns:
            (hint_generator, tokenizer)
        """
        if device is None:
            device = cls.resolve_device(prefer_gpu=prefer_gpu)

        source = model_id
        if model_dir is not None:
            model_dir = Path(model_dir)
            if model_dir.exists():
                source = str(model_dir)
            else:
                logger.warning(
                    "Hint generator dir not found at %s. Using base model %s.",
                    model_dir,
                    model_id,
                )

        logger.info("Loading hint generator from %s ...", source)
        model = AutoModelForSeq2SeqLM.from_pretrained(source)
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(source)
        logger.info("Hint generator loaded on %s.", device)
        return model, tokenizer

    # ------------------------------------------------------------------
    # Tokenizer helper
    # ------------------------------------------------------------------

    @staticmethod
    def get_detector_tokenizer(
        model_id: str = "microsoft/codebert-base",
    ) -> PreTrainedTokenizer:
        """
        Load the CodeBERT tokenizer standalone.
        Useful in data pipeline scripts that need to tokenise without
        loading the full model.
        """
        return AutoTokenizer.from_pretrained(model_id)

    # ------------------------------------------------------------------
    # Config-driven loading
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        model_config: dict,
        device: Optional[torch.device] = None,
    ) -> ModelBundle:
        """
        Build a ModelBundle from a parsed model_config.yaml dictionary.

        Expected config keys:
            checkpoint_path     (optional)
            detector_dir        (optional)
            explanation_dir     (optional)
            hint_dir            (optional)
            explanation_model_id (optional)
            hint_model_id        (optional)
            prefer_gpu           (bool, default True)

        If checkpoint_path is present, from_checkpoint() is called.
        Otherwise, individual models are loaded and bundled manually.
        """
        prefer_gpu = model_config.get("prefer_gpu", True)
        if device is None:
            device = cls.resolve_device(prefer_gpu=prefer_gpu)

        if "checkpoint_path" in model_config:
            return cls.from_checkpoint(
                checkpoint_path=model_config["checkpoint_path"],
                device=device,
            )

        # Load individually and assemble bundle
        registry = ErrorLabelRegistry()

        detector, detector_tok, _ = cls.load_error_detector(
            model_dir=model_config.get("detector_dir", DEFAULT_DETECTOR_DIR),
            device=device,
        )

        explanation_model, explanation_tok = cls.load_explanation_model(
            model_dir=model_config.get("explanation_dir", DEFAULT_EXPLANATION_DIR),
            model_id=model_config.get("explanation_model_id", DEFAULT_EXPLANATION_MODEL_ID),
            device=device,
        )

        hint_gen, hint_tok = cls.load_hint_generator(
            model_dir=model_config.get("hint_dir"),
            model_id=model_config.get("hint_model_id", DEFAULT_HINT_MODEL_ID),
            device=device,
        )

        return ModelBundle(
            error_detector=detector,
            detector_tokenizer=detector_tok,
            explanation_model=explanation_model,
            explanation_tokenizer=explanation_tok,
            hint_generator=hint_gen,
            hint_tokenizer=hint_tok,
            label_registry=registry,
            device=device,
        )
