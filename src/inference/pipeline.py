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
from typing import Any, Iterable, Literal, Optional

import json
from dotenv import load_dotenv
from groq import AsyncGroq
try:

    from src.inference.context_builder import ContextBuilder, LEVEL_GUIDANCE, RepoContext
except ImportError:
    from src.inference.context_builder import ContextBuilder, RepoContext
    LEVEL_GUIDANCE = {
        "beginner": "Provide foundational step-by-step explanations, avoiding dense jargon and using practical real-world analogies.",
        "intermediate": "Focus on algorithmic efficiency, language idioms, clean patterns, and standard edge-case avoidance.",
        "advanced": "Focus on systems architecture, low-level execution semantics, concurrency performance, and security hardening.",
    }
from src.inference.repo_parser import RepoParser



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
    static_analysis_summary: Optional[dict] = None




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
    Groq-backed coding mentor with deterministic hybrid grounding and AST repository context.

    Use chat() for conversational mentoring and debugger walkthroughs; it automatically invokes
    static analysis grounding when code snippets are provided and synthesizes workspace symbols
    when repo_root or file_path coordinates are supplied.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        temperature: float = 0.25,
        max_tokens: int = 1800,
        context_builder: Optional[ContextBuilder] = None,
        repo_parser: Optional[RepoParser] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self._client = AsyncGroq(api_key=self.api_key) if self.api_key else None
        self.context_builder = context_builder or ContextBuilder()
        self.repo_parser = repo_parser or RepoParser()


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
        repo_root: Optional[str | Path] = None,
        file_path: Optional[str] = None,
        **_kwargs: Any,
    ) -> MentorResponse:
        started = time.perf_counter()
        language = _normalize_language(language)
        level_value = _normalize_level(level)

        if (repo_root or file_path) and not repo:
            repo = self.repo_parser.extract_repo_context(
                code=code,
                language=language,
                root_dir=str(repo_root) if repo_root else None,
                file_path=file_path,
            )

        if code.strip() and analysis_result is None:
            try:
                from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer
                analysis_result = AdvancedCodeAnalyzer.analyze(code, language)
            except Exception:
                analysis_result = None

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

        completion = await self._client.chat.completions.create(
            model=self.model,
            messages=packed.messages,
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
            grounded=getattr(packed, "grounding_enabled", False),
            token_utilization_pct=telemetry.get("utilization_pct", 0.0) if isinstance(telemetry, dict) else 0.0,
            truncated_code=getattr(packed, "truncated_code", False),
            dropped_turns=getattr(packed, "dropped_turns", 0),
            static_analysis_summary={
                "total_errors": analysis_result.get("total_errors", 0),
                "quality_score": analysis_result.get("quality_score"),
                "errors": [
                    {
                        "category": e.get("category"),
                        "severity": e.get("severity"),
                        "line": e.get("line"),
                        "message": e.get("message"),
                        "suggestion": e.get("suggestion")
                    }
                    for e in analysis_result.get("errors", [])[:5]
                ]
            } if analysis_result and analysis_result.get("supported", True) else None,
        )



    async def analyze(
        self,
        code: str,
        language: str = DEFAULT_LANGUAGE,
        level: str = DEFAULT_LEVEL,
        question: str = "Review this code, identify likely bugs, teach the concept, and suggest a fix.",
        analysis_result: Optional[dict] = None,
        repo: Optional[RepoContext] = None,
        repo_root: Optional[str | Path] = None,
    ) -> MentorResponse:
        return await self.chat(
            message=question,
            code=code,
            language=language,
            level=level,
            analysis_result=analysis_result,
            repo=repo,
            repo_root=repo_root,
        )

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
            "judgement. Ask a clarifying question if the task is ambiguous.\n\n"
            f"Level behavior: {LEVEL_GUIDANCE[level]}\n\n"
            "You MUST output valid JSON only, using this exact schema:\n"
            "{\n"
            '  "answer": "Your detailed explanation or response.",\n'
            '  "suspected_bugs": ["bug 1", "bug 2"],\n'
            '  "fixed_code": "The corrected code block (if any)",\n'
            '  "lesson": "The main takeaway lesson",\n'
            '  "next_steps": ["step 1", "step 2"]\n'
            "}"
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
