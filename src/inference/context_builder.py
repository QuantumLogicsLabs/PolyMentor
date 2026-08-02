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

    @classmethod
    def format_static_analysis(cls, analysis_result: Optional[dict[str, Any]], max_chars: int = 800) -> str:
        """
        Synthesizes deterministic static analyzer findings into a compact summary for LLM grounding.
        Strictly limits output size to preserve token budget.
        """
        if not analysis_result or not analysis_result.get("supported", True):
            return ""

        total = analysis_result.get("total_errors", 0)
        errors = analysis_result.get("errors", [])
        if total == 0 and not errors:
            return "Static Analysis Grounding: Clean code detected (0 deterministic errors found)."

        lines = [f"Static Analysis Grounding: Detected {total} potential issue(s):"]
        for err in errors[:5]:  # Limit to top 5 most notable issues
            line_num = f"[Line {err['line']}] " if err.get("line") else ""
            sev = err.get("severity", "medium").upper()
            msg = err.get("message", "Issue detected")
            lines.append(f" - {sev} {line_num}{msg}")

        if len(errors) > 5:
            lines.append(f" - ... and {len(errors) - 5} more issue(s). Focus on resolving top errors.")

        summary = "\n".join(lines)
        if len(summary) > max_chars:
            summary = summary[: max_chars - 20] + "... [Truncated]"
        return summary

    @classmethod
    def format_repo_context(cls, repo: Optional[RepoContext]) -> str:
        """
        Formats repository and architectural metadata into concise prompt directives.
        Enables codebase-aware inference without dumping excess files into the context window.
        """
        if not repo:
            return ""

        sections = []
        if repo.file_path or repo.workspace_name:
            location = f"File: {repo.file_path or 'unknown'}"
            if repo.workspace_name:
                location += f" (Workspace: {repo.workspace_name})"
            sections.append(f"Repository Location: {location}")

        if repo.dependencies:
            deps_str = ", ".join(repo.dependencies[:6])
            if len(repo.dependencies) > 6:
                deps_str += "..."
            sections.append(f"Key Dependencies: {deps_str}")

        if repo.git_status:
            sections.append(f"Git Status: {repo.git_status}")

        if repo.related_signatures:
            sigs_str = "; ".join(repo.related_signatures[:3])
            sections.append(f"Related Signatures: {sigs_str}")

        return "\n".join(sections)

    @classmethod
    def pack_history(
        cls,
        history: Optional[Iterable[Any]],
        max_tokens: int = 800,
        max_turns: int = 10,
    ) -> tuple[list[dict[str, str]], int, int]:
        """
        Packs conversational turns from newest to oldest within token budget.
        Returns (packed_messages, used_tokens, dropped_turns_count).
        """
        if not history:
            return [], 0, 0

        raw_turns: list[dict[str, str]] = []
        for item in history:
            if hasattr(item, "role") and hasattr(item, "content"):
                role = getattr(item, "role")
                content = getattr(item, "content")
            elif isinstance(item, dict):
                role = item.get("role", "user")
                content = item.get("content", "")
            else:
                continue

            if role in {"user", "assistant"} and content:
                raw_turns.append({"role": role, "content": str(content)})

        if len(raw_turns) > max_turns:
            dropped_turns = len(raw_turns) - max_turns
            candidate_turns = raw_turns[-max_turns:]
        else:
            dropped_turns = 0
            candidate_turns = raw_turns

        packed_turns: list[dict[str, str]] = []
        accumulated_tokens = 0

        # Traverse backwards from newest to oldest to preserve most recent context
        for turn in reversed(candidate_turns):
            turn_tokens = cls.estimate_tokens(turn["content"]) + 4  # Overhead per message
            if accumulated_tokens + turn_tokens > max_tokens and packed_turns:
                dropped_turns += 1
                continue
            elif accumulated_tokens + turn_tokens <= max_tokens:
                packed_turns.insert(0, turn)
                accumulated_tokens += turn_tokens
            else:
                # Even a single turn exceeds budget, truncate it to fit
                avail_tokens = max(20, max_tokens - accumulated_tokens - 10)
                truncated_text, _ = cls.truncate_code_to_budget(turn["content"], token_budget=avail_tokens)
                packed_turns.insert(0, {"role": turn["role"], "content": truncated_text})
                accumulated_tokens += avail_tokens
                break

        return packed_turns, accumulated_tokens, dropped_turns





