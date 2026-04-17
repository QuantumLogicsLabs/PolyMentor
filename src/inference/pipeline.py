"""
src/inference/pipeline.py
-------------------------
Public API entrypoint for PolyMentor.

This is the ONLY module that external callers should import from.
All other modules in this package are internal implementation details.

Two ways to use this module
----------------------------

1. Python API (direct integration or scripting):

    from src.inference.pipeline import PolyMentorPipeline

    mentor = PolyMentorPipeline.from_pretrained("models_saved/best_mentor_model.pt")

    result = mentor.analyze(
        code=\"\"\"
        for i in range(10):
            if i = 5:
                break
        \"\"\",
        language="python",
        level="beginner",
    )

    print(result.error_type)       # "syntax_error/assignment_in_condition"
    print(result.explanation)      # "You used = which assigns a value..."
    print(result.hint)             # "Step 1: Think about what operator..."
    print(result.concept_taught)   # "Comparison Operators: == vs ="
    print(result.quality_score)    # 72

2. HTTP API (FastAPI server started by bash scripts/run_tutor.sh --api):

    POST /analyze
    {
        "code": "...",
        "language": "python",
        "level": "beginner"
    }

    See /docs for the auto-generated Swagger UI.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.model_factory import ModelBundle, ModelFactory
from src.inference.predict import (
    DetectionResult,
    predict_errors,
    predict_explanation,
    predict_hints,
)
from src.reasoning_engine.error_classifier import ErrorClassifier
from src.reasoning_engine.feedback_scorer import FeedbackScorer
from src.utils.config_loader import load_model_config
from src.utils.logger import setup_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = {"python", "javascript", "cpp", "java"}
SUPPORTED_LEVELS    = {"beginner", "intermediate", "advanced"}

DEFAULT_LEVEL    = "beginner"
DEFAULT_LANGUAGE = "python"
DEFAULT_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# Result dataclass — what analyze() returns
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    """
    The complete output of a single PolyMentor analysis.

    Every field is populated regardless of whether an error was found.
    If the code is clean, error_type is None and quality_score reflects
    only style and complexity, not correctness.

    Attributes
    ----------
    status:
        "error_found" — at least one error was detected.
        "clean"       — no errors found; quality feedback still provided.
        "error"       — an internal failure occurred (check logs).

    error_type:
        Primary detected error type string, e.g.
        "syntax_error/assignment_in_condition". None if status is "clean".

    error_types:
        All detected error types. A snippet can have multiple errors.

    error_location:
        Best-estimate line number of the primary error (1-indexed).
        None if not determinable or if no error found.

    explanation:
        Plain-English explanation of why the primary error occurred and
        what concept it relates to.

    hint:
        The first (most abstract) hint step. Additional hints are in hints[1:].

    hints:
        Ordered list of all generated hints.

    concept_taught:
        The programming concept this error maps to, e.g.
        "Comparison Operators: == vs =".

    quality_score:
        Integer 0–100. Composite of correctness, readability,
        complexity, and clean-code dimensions.

    suggestions:
        Actionable improvement recommendations derived from the quality score.

    confidences:
        Confidence score (0–1) per detected error type.

    language:
        The language the code was analyzed as.

    level:
        The learner level used for explanation depth and hint granularity.

    elapsed_ms:
        Total analysis time in milliseconds.
    """

    status:          str
    error_type:      Optional[str]
    error_types:     List[str]
    error_location:  Optional[int]
    explanation:     str
    hint:            str
    hints:           List[str]
    concept_taught:  str
    quality_score:   int
    suggestions:     List[str]
    confidences:     dict[str, float]
    language:        str
    level:           str
    elapsed_ms:      float


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PolyMentorPipeline:
    """
    Orchestrates the full PolyMentor analysis pipeline end-to-end.

    Initialization
    --------------
    Do not call __init__ directly. Use one of the class method constructors:

        PolyMentorPipeline.from_pretrained(path)   — load from checkpoint file
        PolyMentorPipeline.from_config(config)     — load from config dict
        PolyMentorPipeline.from_bundle(bundle)     — wrap an existing ModelBundle

    Core method
    -----------
        result = pipeline.analyze(code, language, level)

    The pipeline calls these stages in order:
        1. predict_errors()       — error detection (predict.py)
        2. ErrorClassifier        — concept mapping (reasoning_engine)
        3. predict_explanation()  — explanation generation (predict.py)
        4. predict_hints()        — hint generation (predict.py)
        5. FeedbackScorer         — quality scoring (reasoning_engine)
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self._bundle    = bundle
        self._classifier = ErrorClassifier()
        self._scorer     = FeedbackScorer()
        logger.info(
            "PolyMentorPipeline ready. Device: %s", bundle.device
        )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_path: str | Path,
        device: Optional[torch.device] = None,
        prefer_gpu: bool = True,
    ) -> "PolyMentorPipeline":
        """
        Load pipeline from a fused checkpoint file.

        Args:
            checkpoint_path: Path to best_mentor_model.pt.
            device:          Target device. Auto-detected if None.
            prefer_gpu:      Use GPU when available.

        Returns:
            Initialised PolyMentorPipeline ready for inference.

        Example:
            mentor = PolyMentorPipeline.from_pretrained(
                "models_saved/best_mentor_model.pt"
            )
        """
        setup_logger()
        logger.info("Loading PolyMentor from checkpoint: %s", checkpoint_path)
        bundle = ModelFactory.from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            prefer_gpu=prefer_gpu,
        )
        return cls(bundle)

    @classmethod
    def from_config(
        cls,
        config_path: str | Path = "configs/model_config.yaml",
        device: Optional[torch.device] = None,
    ) -> "PolyMentorPipeline":
        """
        Load pipeline using settings from model_config.yaml.

        Args:
            config_path: Path to model_config.yaml or a local override.
            device:      Target device. Auto-detected if None.

        Returns:
            Initialised PolyMentorPipeline.

        Example:
            mentor = PolyMentorPipeline.from_config("configs/model_config.local.yaml")
        """
        setup_logger()
        config = load_model_config(config_path)
        bundle = ModelFactory.from_config(model_config=config, device=device)
        return cls(bundle)

    @classmethod
    def from_bundle(cls, bundle: ModelBundle) -> "PolyMentorPipeline":
        """
        Wrap an already-loaded ModelBundle.

        Useful in training scripts that load models once and want to
        run inference without loading again.
        """
        return cls(bundle)

    # ------------------------------------------------------------------
    # Core analysis method
    # ------------------------------------------------------------------

    def analyze(
        self,
        code: str,
        language: str = DEFAULT_LANGUAGE,
        level: str = DEFAULT_LEVEL,
        threshold: float = DEFAULT_THRESHOLD,
        num_hints: int = 3,
    ) -> AnalysisResult:
        """
        Analyze a code snippet and return the full PolyMentor response.

        This is the main method. It orchestrates all five pipeline stages
        and assembles the final AnalysisResult.

        Args:
            code:       Raw source code to analyze. Any length; will be
                        truncated to the model's maximum input if needed.
            language:   Programming language. One of:
                        "python", "javascript", "cpp", "java".
            level:      Learner level for explanation depth and hint granularity.
                        One of: "beginner", "intermediate", "advanced".
            threshold:  Confidence threshold for error detection (0–1).
                        Lower values catch more errors but increase false positives.
            num_hints:  Number of hint steps to generate.

        Returns:
            AnalysisResult with all fields populated.

        Raises:
            ValueError: If language or level is not supported.
        """
        start_time = time.perf_counter()

        # -----------------------------------------------------------------
        # Input validation
        # -----------------------------------------------------------------
        language = language.lower().strip()
        level    = level.lower().strip()

        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Unsupported language: '{language}'. "
                f"Choose from: {sorted(SUPPORTED_LANGUAGES)}"
            )
        if level not in SUPPORTED_LEVELS:
            raise ValueError(
                f"Unsupported level: '{level}'. "
                f"Choose from: {sorted(SUPPORTED_LEVELS)}"
            )
        if not code or not code.strip():
            return self._empty_result(language, level, "No code provided.")

        # -----------------------------------------------------------------
        # Stage 1 — Error Detection
        # -----------------------------------------------------------------
        logger.debug("Stage 1: Error detection (%s, %s)", language, level)

        detection: DetectionResult = predict_errors(
            code=code,
            language=language,
            model=self._bundle.error_detector,
            tokenizer=self._bundle.detector_tokenizer,
            registry=self._bundle.label_registry,
            device=self._bundle.device,
            threshold=threshold,
        )

        if not detection.has_error:
            # No errors detected — still run quality scoring and return
            logger.debug("No errors detected. Running quality score only.")
            score_result = self._scorer.score(code, language)
            elapsed = (time.perf_counter() - start_time) * 1000

            return AnalysisResult(
                status="clean",
                error_type=None,
                error_types=[],
                error_location=None,
                explanation="No errors detected in this code snippet.",
                hint="Your code looks correct! Review the quality suggestions below.",
                hints=[],
                concept_taught="",
                quality_score=score_result.score,
                suggestions=score_result.suggestions,
                confidences={},
                language=language,
                level=level,
                elapsed_ms=round(elapsed, 2),
            )

        # Use the highest-confidence error as the primary one
        primary_error = max(detection.confidences, key=detection.confidences.get)

        # -----------------------------------------------------------------
        # Stage 2 — Concept Mapping (Reasoning Engine)
        # -----------------------------------------------------------------
        logger.debug("Stage 2: Concept mapping for '%s'", primary_error)

        classification = self._classifier.classify(
            error_label=primary_error,
            language=language,
            level=level,
        )
        concept        = classification.concept
        error_location = classification.estimated_line(code)

        # -----------------------------------------------------------------
        # Stage 3 — Explanation Generation
        # -----------------------------------------------------------------
        logger.debug("Stage 3: Explanation generation")

        explanation_result = predict_explanation(
            code=code,
            language=language,
            error_label=primary_error,
            concept=concept,
            level=level,
            model=self._bundle.explanation_model,
            tokenizer=self._bundle.explanation_tokenizer,
            device=self._bundle.device,
        )

        # -----------------------------------------------------------------
        # Stage 4 — Hint Generation
        # -----------------------------------------------------------------
        logger.debug("Stage 4: Hint generation (%d hints)", num_hints)

        hint_result = predict_hints(
            code=code,
            language=language,
            error_label=primary_error,
            concept=concept,
            level=level,
            model=self._bundle.hint_generator,
            tokenizer=self._bundle.hint_tokenizer,
            device=self._bundle.device,
            num_hints=num_hints,
        )

        # -----------------------------------------------------------------
        # Stage 5 — Quality Scoring
        # -----------------------------------------------------------------
        logger.debug("Stage 5: Quality scoring")

        score_result = self._scorer.score(code, language)

        # -----------------------------------------------------------------
        # Assemble result
        # -----------------------------------------------------------------
        elapsed = (time.perf_counter() - start_time) * 1000
        logger.debug("Analysis complete in %.1f ms", elapsed)

        return AnalysisResult(
            status="error_found",
            error_type=primary_error,
            error_types=detection.error_labels,
            error_location=error_location,
            explanation=explanation_result.explanation,
            hint=hint_result.hints[0] if hint_result.hints else "",
            hints=hint_result.hints,
            concept_taught=classification.concept_display_name,
            quality_score=score_result.score,
            suggestions=score_result.suggestions,
            confidences=detection.confidences,
            language=language,
            level=level,
            elapsed_ms=round(elapsed, 2),
        )

    # ------------------------------------------------------------------
    # Convenience method — analyze multiple snippets
    # ------------------------------------------------------------------

    def analyze_batch(
        self,
        snippets: List[dict],
        default_language: str = DEFAULT_LANGUAGE,
        default_level: str = DEFAULT_LEVEL,
    ) -> List[AnalysisResult]:
        """
        Analyze a list of code snippets.

        Each item in snippets is a dict with keys:
            "code"     (required)
            "language" (optional, defaults to default_language)
            "level"    (optional, defaults to default_level)

        Returns a list of AnalysisResult in the same order.

        Example:
            results = mentor.analyze_batch([
                {"code": "if x = 5: pass", "language": "python"},
                {"code": "var x = ;",      "language": "javascript"},
            ])
        """
        return [
            self.analyze(
                code=item["code"],
                language=item.get("language", default_language),
                level=item.get("level", default_level),
            )
            for item in snippets
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _empty_result(
        self, language: str, level: str, message: str
    ) -> AnalysisResult:
        return AnalysisResult(
            status="clean",
            error_type=None,
            error_types=[],
            error_location=None,
            explanation=message,
            hint="",
            hints=[],
            concept_taught="",
            quality_score=0,
            suggestions=[],
            confidences={},
            language=language,
            level=level,
            elapsed_ms=0.0,
        )


# ===========================================================================
# FastAPI HTTP Application
# ===========================================================================
# Instantiated when bash scripts/run_tutor.sh --api is called.
# The pipeline is loaded once at startup from the environment variable
# POLYMENTOR_MODEL_PATH (set by run_tutor.sh).
# ===========================================================================

app = FastAPI(
    title="PolyMentor API",
    description=(
        "AI-powered coding mentor. Submit code in any supported language "
        "and receive error explanations, hints, and quality feedback."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance — populated on startup
_pipeline: Optional[PolyMentorPipeline] = None


@app.on_event("startup")
async def _startup() -> None:
    global _pipeline
    checkpoint = os.environ.get(
        "POLYMENTOR_MODEL_PATH", "models_saved/best_mentor_model.pt"
    )
    logger.info("API startup: loading model from %s", checkpoint)
    _pipeline = PolyMentorPipeline.from_pretrained(checkpoint)
    logger.info("API ready.")


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    code:     str  = Field(..., description="Source code to analyze.")
    language: str  = Field(DEFAULT_LANGUAGE, description="python | javascript | cpp | java")
    level:    str  = Field(DEFAULT_LEVEL, description="beginner | intermediate | advanced")
    threshold: float = Field(DEFAULT_THRESHOLD, ge=0.0, le=1.0)
    num_hints: int   = Field(3, ge=1, le=5)
    session_id: Optional[str] = Field(None, description="Opaque session identifier.")


class AnalyzeResponse(BaseModel):
    status:         str
    error_type:     Optional[str]
    error_types:    List[str]
    error_location: Optional[int]
    explanation:    str
    hint:           str
    hints:          List[str]
    concept_taught: str
    quality_score:  int
    suggestions:    List[str]
    confidences:    dict
    language:       str
    level:          str
    elapsed_ms:     float


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Health check. Returns 200 if the API is running and model is loaded."""
    return {
        "status": "ok",
        "model_loaded": _pipeline is not None,
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a code snippet and return error explanations, hints, and quality feedback.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")

    try:
        result = _pipeline.analyze(
            code=request.code,
            language=request.language,
            level=request.level,
            threshold=request.threshold,
            num_hints=request.num_hints,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error during analysis.")
        raise HTTPException(status_code=500, detail="Internal analysis error.") from exc

    return AnalyzeResponse(
        status=result.status,
        error_type=result.error_type,
        error_types=result.error_types,
        error_location=result.error_location,
        explanation=result.explanation,
        hint=result.hint,
        hints=result.hints,
        concept_taught=result.concept_taught,
        quality_score=result.quality_score,
        suggestions=result.suggestions,
        confidences=result.confidences,
        language=result.language,
        level=result.level,
        elapsed_ms=result.elapsed_ms,
    )
