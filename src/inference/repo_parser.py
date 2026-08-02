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
from typing import Any, Optional


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

    def extract_symbols(self, code: str, language: str) -> dict[str, list[str]]:
        """
        Extract classes, functions, and import statements from code using Tree-sitter AST parsing,
        falling back to regular expressions if necessary.
        """
        results: dict[str, list[str]] = {"classes": [], "functions": [], "imports": []}
        if not code or not code.strip():
            return results

        tree = self.parse_code(code, language)
        if tree and tree.root_node:
            try:
                self._walk_ast_symbols(tree.root_node, results)
                return results
            except Exception as e:
                logger.debug(f"AST symbol extraction error: {e}. Falling back to regex.")

        return self._extract_heuristic_symbols(code, self.normalize_lang(language))

    def _walk_ast_symbols(self, node: Any, results: dict[str, list[str]]) -> None:
        """Recursively traverse Tree-sitter AST nodes to collect structural code symbols."""
        node_type = node.type
        text = node.text.decode("utf-8", errors="replace") if hasattr(node, "text") and node.text else ""

        # Check classes
        if node_type in ("class_definition", "class_declaration", "class_specifier"):
            for child in node.children:
                if child.type in ("identifier", "type_identifier"):
                    name = child.text.decode("utf-8", errors="replace") if child.text else ""
                    if name and name not in results["classes"]:
                        results["classes"].append(name)
                    break
            else:
                # Fallback to first line of class definition
                first_line = text.split("\n")[0].split("{")[0].split(":")[0].strip()
                if first_line and first_line not in results["classes"]:
                    results["classes"].append(first_line)

        # Check functions & methods
        elif node_type in ("function_definition", "function_declaration", "method_definition", "method_declaration"):
            for child in node.children:
                if child.type in ("identifier", "field_identifier"):
                    name = child.text.decode("utf-8", errors="replace") if child.text else ""
                    if name and name not in results["functions"]:
                        results["functions"].append(name)
                    break
            else:
                first_line = text.split("\n")[0].split("{")[0].split(":")[0].strip()
                if first_line and first_line not in results["functions"]:
                    results["functions"].append(first_line)

        # Check imports
        elif node_type in ("import_statement", "import_from_statement", "import_declaration", "preproc_include"):
            line = text.split("\n")[0].strip().rstrip(";")
            if line and line not in results["imports"]:
                results["imports"].append(line)

        for child in node.children:
            self._walk_ast_symbols(child, results)

    def _extract_heuristic_symbols(self, code: str, lang: str) -> dict[str, list[str]]:
        """Regex-based heuristic fallback for symbol extraction when AST grammars are unavailable."""
        results: dict[str, list[str]] = {"classes": [], "functions": [], "imports": []}
        lines = code.split("\n")

        for line in lines:
            stripped = line.strip()
            # Imports
            if stripped.startswith("import ") or stripped.startswith("from ") or stripped.startswith("#include") or stripped.startswith("import {"):
                val = stripped.rstrip(";")
                if val not in results["imports"]:
                    results["imports"].append(val)
            # Classes
            if re.match(r"^class\s+([a-zA-Z0-9_]+)", stripped):
                m = re.match(r"^class\s+([a-zA-Z0-9_]+)", stripped)
                if m and m.group(1) not in results["classes"]:
                    results["classes"].append(m.group(1))
            # Functions
            m_func = re.match(r"^(?:def|function|async\s+def|async\s+function|void|int|double|string|bool)\s+([a-zA-Z0-9_]+)\s*\(", stripped)
            if m_func and m_func.group(1) not in results["functions"]:
                results["functions"].append(m_func.group(1))
        
        return results

