"""
src/inference/context_builder.py
--------------------------------
Intelligent context builder for efficient prompt packing and hybrid AI mentoring.
Aggregates code analysis findings, learner history, repository context, and skill level
guidance into optimized token budgets for Groq LLM inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Optional

LearnerLevel = Literal["beginner", "intermediate", "advanced"]

DEFAULT_TOKEN_BUDGET = 3500  # Conservative budget for fast free tier prompt packing
ESTIMATED_CHARS_PER_TOKEN = 3.8

LEVEL_GUIDANCE: dict[LearnerLevel, str] = {
    "beginner": (
        "Explain in beginner-friendly language. Use short sentences, define any "
        "technical term before using it, and prefer tiny examples. Do not jump "
        "to advanced patterns, clever shortcuts, complex architecture, or "
        "performance-heavy solutions unless the user explicitly asks for them."
    ),
    "intermediate": (
        "Assume the learner knows basic syntax. Explain the reasoning and tradeoffs, "
        "but keep the solution practical and avoid advanced architecture or clever "
        "optimizations unless they are necessary for the question."
    ),
    "advanced": (
        "You may use precise technical language and discuss deeper tradeoffs, but "
        "still keep the answer focused on the user's question and avoid unnecessary "
        "over-engineering."
    ),
}


@dataclass
class RepoContext:
    """Represents file architecture and surrounding workspace context."""

    file_path: Optional[str] = None
    workspace_name: Optional[str] = None
    dependencies: list[str] = field(default_factory=list)
    git_status: Optional[str] = None
    related_signatures: list[str] = field(default_factory=list)


@dataclass
class PackedPrompt:
    """Result of context packing, containing formatted system and user messages."""

    messages: list[dict[str, str]]
    estimated_tokens: int
    truncated_code: bool = False
    dropped_turns: int = 0
    grounding_enabled: bool = False
