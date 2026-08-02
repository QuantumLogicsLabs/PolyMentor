"""
src/inference/repo_parser.py
----------------------------
Tree-sitter repository and source code parser for PolyMentor.
Provides deterministic AST symbol extraction (classes, functions, imports) and
workspace dependency discovery to ground LLM reasoning with accurate repository context.
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import tree_sitter
    from tree_sitter import Language, Parser
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning("tree-sitter module not found; falling back to heuristic symbol analysis.")

try:
    import tree_sitter_python
except ImportError:
    tree_sitter_python = None

try:
    import tree_sitter_javascript
except ImportError:
    tree_sitter_javascript = None

try:
    import tree_sitter_cpp
except ImportError:
    tree_sitter_cpp = None

try:
    import tree_sitter_java
except ImportError:
    tree_sitter_java = None


class RepoParser:
    """
    Analyzes source files and project repositories using Tree-sitter AST parsing.
    Extracts structural symbols and identifies inter-file dependencies in a workspace.
    """

    def __init__(self) -> None:
        self._parsers: dict[str, Parser] = {}
        self._languages: dict[str, Language] = {}
        self.tree_sitter_ready = False
        if TREE_SITTER_AVAILABLE:
            self._init_parsers()

    def _init_parsers(self) -> None:
        """Initializes Tree-sitter language grammars for supported programming languages."""
        grammar_map = {
            "python": tree_sitter_python,
            "javascript": tree_sitter_javascript,
            "cpp": tree_sitter_cpp,
            "java": tree_sitter_java,
        }
        for lang_name, mod in grammar_map.items():
            if mod and hasattr(mod, "language"):
                try:
                    lang = Language(mod.language())
                    parser = Parser(lang)
                    self._languages[lang_name] = lang
                    self._parsers[lang_name] = parser
                except Exception as e:
                    logger.debug(f"Failed to load Tree-sitter grammar for {lang_name}: {e}")
        if self._parsers:
            self.tree_sitter_ready = True
            logger.info(f"Initialized Tree-sitter AST parsers for: {list(self._parsers.keys())}")

    def normalize_lang(self, language: str) -> str:
        """Normalize common language alias strings to canonical Tree-sitter grammar names."""
        lower = (language or "python").lower().strip()
        if lower in ("py", "python3"):
            return "python"
        if lower in ("js", "jsx", "node", "typescript", "ts", "javascript"):
            return "javascript"
        if lower in ("c++", "cc", "cxx", "c", "cpp"):
            return "cpp"
        if lower in ("java",):
            return "java"
        return lower

    def parse_code(self, code: str, language: str) -> Optional[tree_sitter.Tree]:
        """Parse source code string into a Tree-sitter AST if grammar is available."""
        if not self.tree_sitter_ready or not code:
            return None
        lang = self.normalize_lang(language)
        parser = self._parsers.get(lang)
        if not parser:
            return None
        try:
            return parser.parse(code.encode("utf-8", errors="replace"))
        except Exception as e:
            logger.warning(f"Error parsing {language} code AST: {e}")
            return None
