"""
src/inference/pipeline.py
-------------------------
Public context-aware inference entrypoint for PolyMentor.

PolyMentor is an AI coding tutor and mentor powered by Groq and static analysis grounding.
It seamlessly incorporates multi-turn conversational history, skill level pedagogical adaptation
(beginner, intermediate, advanced), multi-language normalization, and code token budgeting.

Environment:
    GROQ_API_KEY   Required for Groq responses.
    GROQ_MODEL     Optional. Defaults to llama-3.3-70b-versatile.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import json
from dotenv import load_dotenv
from groq import AsyncGroq
from src.inference.context_builder import ContextBuilder, RepoContext, PackedPrompt

load_dotenv()

__all__ = [
    "LearnerLevel",
    "ChatMessage",
    "MentorResponse",
    "PolyMentorPipeline",
    "DEFAULT_MODEL",
    "DEFAULT_LEVEL",
    "DEFAULT_LANGUAGE",
]



SUPPORTED_LANGUAGES = {
    "python",
    "javascript",
    "typescript",
    "java",
    "cpp",
    "c",
    "csharp",
    "go",
    "rust",
    "php",
    "ruby",
    "swift",
    "kotlin",
    "sql",
    "html",
    "css",
}

LearnerLevel = Literal["beginner", "intermediate", "advanced"]

DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
DEFAULT_LEVEL: LearnerLevel = "beginner"
DEFAULT_LANGUAGE = "python"

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
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass
class MentorResponse:
    status: str
    answer: str
    language: str
    level: LearnerLevel
    model: str
    suspected_bugs: list[str] = field(default_factory=list)
    fixed_code: Optional[str] = None
    lesson: Optional[str] = None
    next_steps: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    grounded: bool = False
    token_utilization_pct: float = 0.0
    truncated_code: bool = False
    dropped_turns: int = 0



def _normalize_language(language: str | None) -> str:
    value = (language or DEFAULT_LANGUAGE).strip().lower()
    aliases = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "c++": "cpp",
        "c#": "csharp",
    }
    return aliases.get(value, value)


def _normalize_level(level: str | None) -> LearnerLevel:
    value = (level or DEFAULT_LEVEL).strip().lower()
    if value in {"beginner", "intermediate", "advanced"}:
        return value  # type: ignore[return-value]
    return DEFAULT_LEVEL


class PolyMentorPipeline:
    """
    Groq-backed coding mentor.

    Use chat() for normal chatbot turns and analyze() when you have code plus a
    debugging/teaching question. analyze() is kept as a compatibility alias for
    older integrations, but it now returns a MentorResponse instead of scores.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 1800,
        context_builder: Optional[ContextBuilder] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=self.api_key) if self.api_key else None
        self.context_builder = context_builder or ContextBuilder()


    @classmethod
    def from_groq(
        cls,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ) -> "PolyMentorPipeline":
        return cls(model=model, api_key=api_key)

    @classmethod
    def from_pretrained(cls, *_args, **_kwargs) -> "PolyMentorPipeline":
        """Compatibility constructor for old checkpoint-based callers."""
        return cls.from_groq()

    async def chat(
        self,
        message: str,
        code: str = "",
        language: str = DEFAULT_LANGUAGE,
        level: str = DEFAULT_LEVEL,
        history: Optional[Iterable[ChatMessage | dict[str, str]]] = None,
        analysis_result: Optional[dict] = None,
        repo: Optional[RepoContext] = None,
    ) -> MentorResponse:
        started = time.perf_counter()
        language = _normalize_language(language)
        level_value = _normalize_level(level)

        if not self._client:
            return MentorResponse(
                status="missing_groq_api_key",
                answer=(
                    "GROQ_API_KEY is not set. Add it to your environment, then "
                    "restart PolyMentor. Example: export GROQ_API_KEY='your_key'."
                ),
                language=language,
                level=level_value,
                model=self.model,
                next_steps=[
                    "Create a Groq API key in the Groq console.",
                    "Set GROQ_API_KEY in your shell or deployment environment.",
                    "Run the tutor or API again.",
                ],
                elapsed_ms=(time.perf_counter() - started) * 1000,
            )

        packed = self.context_builder.build_prompt(
            message=message,
            code=code,
            language=language,
            level=level_value,
            history=history,
            analysis_result=analysis_result,
            repo=repo,
            require_json=True,
        )
        messages = packed.messages
        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
            response_format={"type": "json_object"},
        )

        content = completion.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"answer": content}

        telemetry = self.context_builder.inspect_prompt_budget(packed)
        return MentorResponse(
            status="ok",
            answer=parsed.get("answer", ""),
            language=language,
            level=level_value,
            model=self.model,
            suspected_bugs=parsed.get("suspected_bugs", []),
            fixed_code=parsed.get("fixed_code"),
            lesson=parsed.get("lesson"),
            next_steps=parsed.get("next_steps", []),
            elapsed_ms=(time.perf_counter() - started) * 1000,
            grounded=packed.grounding_enabled,
            token_utilization_pct=telemetry["utilization_pct"],
            truncated_code=packed.truncated_code,
            dropped_turns=packed.dropped_turns,
        )


    async def analyze(
        self,
        code: str,
        language: str = DEFAULT_LANGUAGE,
        level: str = DEFAULT_LEVEL,
        question: str = "Review this code, identify likely bugs, teach the concept, and suggest a fix.",
        analysis_result: Optional[dict] = None,
        repo: Optional[RepoContext] = None,
    ) -> MentorResponse:
        return await self.chat(
            message=question,
            code=code,
            language=language,
            level=level,
            analysis_result=analysis_result,
            repo=repo,
        )

    def _build_messages(
        self,
        message: str,
        code: str,
        language: str,
        level: LearnerLevel,
        history: Optional[Iterable[ChatMessage | dict[str, str]]],
        analysis_result: Optional[dict] = None,
        repo: Optional[RepoContext] = None,
    ) -> list[dict[str, str]]:
        packed = self.context_builder.build_prompt(
            message=message,
            code=code,
            language=language,
            level=level,
            history=history,
            analysis_result=analysis_result,
            repo=repo,
            require_json=True,
        )
        return packed.messages

