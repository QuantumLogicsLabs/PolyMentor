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


class ContextBuilder:
    """
    Constructs optimized prompt payloads for PolyMentor chat and analysis workflows.
    Handles token budget enforcement and deterministic context merging.
    """

    def __init__(self, max_tokens: int = DEFAULT_TOKEN_BUDGET) -> None:
        self.max_tokens = max_tokens

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimates the token count of a given string using standard char-to-token ratio.
        Provides fast, deterministic approximation without requiring network or heavy tokenizers.
        """
        if not text:
            return 0
        return max(1, math.ceil(len(text) / ESTIMATED_CHARS_PER_TOKEN))

    @classmethod
    def truncate_code_to_budget(cls, code: str, token_budget: int = 1500) -> tuple[str, bool]:
        """
        Intelligently truncates source code if it exceeds the available token budget.
        Retains both head and tail sections to preserve imports, function headers, and trailing logic.
        Returns a tuple of (processed_code, was_truncated).
        """
        if not code or cls.estimate_tokens(code) <= token_budget:
            return code, False

        max_chars = int(token_budget * ESTIMATED_CHARS_PER_TOKEN)
        if max_chars <= 100:
            return code[:max_chars] + "\n... [Truncated]", True

        head_chars = max_chars // 2 - 50
        tail_chars = max_chars - head_chars - 100
        head_text = code[:head_chars]
        tail_text = code[-tail_chars:] if tail_chars > 0 else ""

        trimmed_lines = code[head_chars:-tail_chars].count("\n") if tail_chars > 0 else code[head_chars:].count("\n")
        marker = f"\n... [Code middle-truncated for prompt efficiency: ~{trimmed_lines} lines trimmed] ...\n"

        return head_text + marker + tail_text, True


