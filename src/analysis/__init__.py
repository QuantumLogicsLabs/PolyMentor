"""
src/analysis/__init__.py
------------------------
Deterministic static code analysis module for PolyMentor.
Provides multi-language bug detection, code quality scoring, and verified fix suggestions
to ground AI responses and eliminate hallucinations in hybrid conversational mentoring.
"""


from .advanced_analyzer import (
    AdvancedCodeAnalyzer,
    CodeError,
    ErrorSeverity,
    ErrorCategory,
    PythonAnalyzer,
    JavaScriptAnalyzer,
    CPPAnalyzer,
    JavaAnalyzer,
)

__all__ = [
    "AdvancedCodeAnalyzer",
    "CodeError",
    "ErrorSeverity",
    "ErrorCategory",
    "PythonAnalyzer",
    "JavaScriptAnalyzer",
    "CPPAnalyzer",
    "JavaAnalyzer",
]
