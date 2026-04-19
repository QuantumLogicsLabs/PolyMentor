from .hint_strategies import get_hint_strategy, DEFAULT_HINTS


class HintSystem:
    """Generates progressive, step-by-step hints for a detected error."""

    def generate_hints(self, error_type: str, code: str, context: dict = None) -> list:
        context = context or {}
        level = context.get("level", "beginner")
        
        # Keep track of error label for the fallback strategy mapping
        context["error_label"] = error_type
        
        strategy = get_hint_strategy(error_type)
        hints = strategy.generate(code, context)

        # For advanced users, return fewer leading hints
        if level == "advanced":
            return hints[-1:]
        elif level == "intermediate":
            return hints[1:]
        return hints  # beginner gets all steps

    def get_hints(self, error_label: str, level: str = "beginner") -> list:
        # route old method → new system
        return self.generate_hints(error_label, code="", context={"level": level})

    def get_first_hint(self, error_label: str, level: str = "beginner") -> str:
        hints = self.get_hints(error_label, level)
        return hints[0] if hints else DEFAULT_HINTS[0]
