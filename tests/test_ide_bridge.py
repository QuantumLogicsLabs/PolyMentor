"""
tests/test_ide_bridge.py
------------------------
Unit tests for IDE integration bridge, LSP diagnostic formatting, and severity mapping.
"""

import unittest
import sys
import os
import json

sys.path.insert(0, os.path.abspath("."))

from scripts.ide_bridge import (
    map_severity_to_lsp,
    convert_error_to_lsp_diagnostic,
    infer_language_from_name,
    analyze_buffer
)
from src.analysis.advanced_analyzer import ErrorSeverity


class TestIDEBridge(unittest.TestCase):
    def test_severity_mapping(self):
        self.assertEqual(map_severity_to_lsp("CRITICAL"), 1)
        self.assertEqual(map_severity_to_lsp("ErrorSeverity.HIGH"), 2)
        self.assertEqual(map_severity_to_lsp("MEDIUM"), 3)
        self.assertEqual(map_severity_to_lsp("LOW"), 4)

    def test_language_inference(self):
        self.assertEqual(infer_language_from_name("src/components/App.tsx"), "javascript")
        self.assertEqual(infer_language_from_name("main.cpp"), "cpp")
        self.assertEqual(infer_language_from_name("python"), "python")

    def test_lsp_diagnostic_zero_indexing(self):
        mock_error = {
            "line": 10,
            "column": 4,
            "message": "Unused variable detected",
            "suggestion": "Remove unused variable 'x'",
            "severity": "HIGH",
            "category": "clean_code"
        }
        diag = convert_error_to_lsp_diagnostic(mock_error)
        self.assertEqual(diag["range"]["start"]["line"], 9)  # 0-indexed: line 10 -> 9
        self.assertEqual(diag["range"]["start"]["character"], 4)
        self.assertEqual(diag["severity"], 2)
        self.assertIn("💡 Refactor Advice:", diag["message"])

    def test_analyze_buffer_lsp_format(self):
        js_code = "function foo(x) { if (x = 5) { console.log('oops'); } }"
        payload = analyze_buffer(js_code, "test.js", format_mode="lsp")
        self.assertEqual(payload["language"], "javascript")
        self.assertGreaterEqual(payload["total_issues"], 1)
        self.assertIsInstance(payload["diagnostics"], list)
        self.assertIn("range", payload["diagnostics"][0])


if __name__ == "__main__":
    unittest.main()
