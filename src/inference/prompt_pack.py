"""
src/inference/prompt_pack.py
----------------------------
Manages prompt templates and construction for PolyMentor.
Cleanly separates prompt design from pipeline execution logic.
"""

from typing import Iterable, Optional, Dict, List, Union
from dataclasses import dataclass

# In Python 3.10+, we can use TypeAlias, but for compatibility we'll use Union
ChatMessageDef = Union[Dict[str, str], any] # 'any' to allow ChatMessage object from pipeline

LEVEL_GUIDANCE = {
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

class PromptPack:
    """Constructs prompt messages for the Groq inference pipeline."""

    @staticmethod
    def build_messages(
        message: str,
        code: str,
        language: str,
        level: str,
        history: Optional[Iterable[ChatMessageDef]] = None,
        analyzer_context: str = ""
    ) -> List[Dict[str, str]]:
        """
        Builds the complete message history for the chat completion.
        Injects level guidance and optional analyzer context.
        """
        system = (
            "You are PolyMentor, a coding tutor chatbot. Your job is to teach "
            "programming, help users write code, identify likely bugs, explain "
            "why bugs happen, and guide learners across many programming "
            "languages. Be practical, friendly, and precise. Do not produce a "
            "numeric quality score. Prefer teaching and corrected examples over "
            "judgement. Ask a clarifying question if the task is ambiguous.\n\n"
            f"Level behavior: {LEVEL_GUIDANCE.get(level, LEVEL_GUIDANCE['beginner'])}\n\n"
            "You MUST output valid JSON only, using this exact schema:\n"
            "{\n"
            '  "answer": "Your detailed explanation or response.",\n'
            '  "suspected_bugs": ["bug 1", "bug 2"],\n'
            '  "fixed_code": "The corrected code block (if any)",\n'
            '  "lesson": "The main takeaway lesson",\n'
            '  "next_steps": ["step 1", "step 2"]\n'
            "}"
        )

        user_content = [
            f"Learner level: {level}",
            f"Language: {language}",
        ]
        
        if analyzer_context:
            user_content.append(f"\n[Static Analysis Results]\n{analyzer_context}\n")
            
        user_content.append(f"User request: {message.strip()}")

        if code.strip():
            user_content.append(f"\nCode:\n```{language}\n{code.strip()}\n```")

        user_prompt = "\n".join(user_content)

        messages: List[Dict[str, str]] = [{"role": "system", "content": system}]
        
        if history:
            for item in history:
                if isinstance(item, dict):
                    role = item.get("role", "user")
                    content = item.get("content", "")
                else:
                    role = getattr(item, "role", "user")
                    content = getattr(item, "content", "")
                    
                if role in {"user", "assistant"} and content:
                    messages.append({"role": role, "content": content})
                    
        messages.append({"role": "user", "content": user_prompt})
        return messages
