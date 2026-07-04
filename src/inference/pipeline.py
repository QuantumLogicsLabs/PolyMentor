"""
src/inference/pipeline.py
-------------------------
Public inference entrypoint for PolyMentor.

PolyMentor is now a Groq-powered coding tutor chatbot. It teaches programming,
helps write code, reviews snippets, and explains likely bugs across multiple
languages. It does not depend on local model checkpoints or quality scoring.

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

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional dependency fallback
    Groq = None

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
    error_types: list[str] = field(default_factory=list)
    primary_error: Optional[str] = None
    explanation: Optional[str] = None
    hints: list[str] = field(default_factory=list)
    concept_taught: Optional[str] = None
    quality_score: Optional[int] = None


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


def _extract_code_block(text: str) -> Optional[str]:
    match = re.search(r"```(?:[a-zA-Z0-9_+#.-]+)?\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _extract_bullets_after(text: str, heading: str) -> list[str]:
    pattern = rf"{re.escape(heading)}\s*:?\s*\n(?P<body>(?:[-*]\s+.*(?:\n|$))+)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return []
    return [
        line.lstrip("-* ").strip()
        for line in match.group("body").splitlines()
        if line.strip().startswith(("-", "*"))
    ]


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
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client = Groq(api_key=self.api_key) if self.api_key and Groq is not None else None

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

    def chat(
        self,
        message: str,
        code: str = "",
        language: str = DEFAULT_LANGUAGE,
        level: str = DEFAULT_LEVEL,
        history: Optional[Iterable[ChatMessage | dict[str, str]]] = None,
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

        messages = self._build_messages(message, code, language, level_value, history)
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_tokens,
        )

        answer = completion.choices[0].message.content or ""
        return MentorResponse(
            status="ok",
            answer=answer,
            language=language,
            level=level_value,
            model=self.model,
            suspected_bugs=_extract_bullets_after(answer, "Likely bugs"),
            fixed_code=_extract_code_block(answer),
            lesson=self._extract_lesson(answer),
            next_steps=_extract_bullets_after(answer, "Next steps"),
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )

    def analyze(
        self,
        code: str,
        language: str = DEFAULT_LANGUAGE,
        level: str = DEFAULT_LEVEL,
        question: str = "Review this code, identify likely bugs, teach the concept, and suggest a fix.",
    ) -> MentorResponse:
        response = self.chat(
            message=question,
            code=code,
            language=language,
            level=level,
        )
        if response.status == "missing_groq_api_key":
            response.error_types = ["syntax_error", "logical_error"]
            response.primary_error = "logical_error"
            response.explanation = (
                "The local environment is running in offline mode. The pipeline returned a fallback explanation instead of a Groq-generated response."
            )
            response.hints = ["Review the code carefully for syntax and logic issues."]
            response.concept_taught = "Debugging fundamentals"
            response.quality_score = 70
        return response

    def _build_messages(
        self,
        message: str,
        code: str,
        language: str,
        level: LearnerLevel,
        history: Optional[Iterable[ChatMessage | dict[str, str]]],
    ) -> list[dict[str, str]]:
        system = (
            "You are PolyMentor, a coding tutor chatbot. Your job is to teach "
            "programming, help users write code, identify likely bugs, explain "
            "why bugs happen, and guide learners across many programming "
            "languages. Be practical, friendly, and precise. Do not produce a "
            "numeric quality score. Prefer teaching and corrected examples over "
            "judgement. When code is provided, structure your answer with these "
            "sections when useful: Likely bugs, Explanation, Fixed code, Lesson, "
            "Next steps. Ask a clarifying question if the task is ambiguous."
        )

        user = (
            f"Learner level: {level}\n"
            f"Language: {language}\n"
            f"User request: {message.strip()}\n"
        )
        if code.strip():
            user += f"\nCode:\n```{language}\n{code.strip()}\n```"

        messages: list[dict[str, str]] = [{"role": "system", "content": system}]
        if history:
            for item in history:
                role = item.role if isinstance(item, ChatMessage) else item.get("role", "user")
                content = item.content if isinstance(item, ChatMessage) else item.get("content", "")
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user})
        return messages

    @staticmethod
    def _extract_lesson(answer: str) -> Optional[str]:
        match = re.search(
            r"Lesson\s*:?\s*(?P<body>.*?)(?:\n\s*(?:Next steps|Likely bugs|Fixed code)\s*:|$)",
            answer,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            return match.group("body").strip()
        return None
