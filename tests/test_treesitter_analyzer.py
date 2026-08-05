"""
tests/test_treesitter_analyzer.py
---------------------------------
Unit test suite verifying Tree-sitter integration in AdvancedCodeAnalyzer and RepoParser.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.abspath("."))

from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer
from src.inference.repo_parser import RepoParser


class TestTreeSitterAnalyzer(unittest.TestCase):
    def setUp(self):
        self.parser = RepoParser()

    def test_repo_parser_symbols_python(self):
        py_code = """
class MyClass:
    def my_func(self):
        pass
"""
        symbols = self.parser.extract_symbols(py_code, "python")
        self.assertIn("MyClass", symbols["classes"])
        self.assertIn("my_func", symbols["functions"])

    def test_syntax_error_detection_javascript(self):
        js_code = "function test() { console.log('unclosed"
        res = AdvancedCodeAnalyzer.analyze(js_code, "javascript")
        self.assertGreaterEqual(res["total_errors"], 1)
        self.assertEqual(res["supported"], True)

    def test_syntax_error_detection_python(self):
        py_code = "def foo(\n    print('invalid syntax')"
        res = AdvancedCodeAnalyzer.analyze(py_code, "python")
        self.assertGreaterEqual(res["total_errors"], 1)
        self.assertGreater(res["critical_count"], 0)

    def test_syntax_error_detection_cpp(self):
        cpp_code = "int main() { return 0;"
        res = AdvancedCodeAnalyzer.analyze(cpp_code, "cpp")
        self.assertGreaterEqual(res["total_errors"], 1)

    def test_syntax_error_detection_java(self):
        java_code = "public class Hello { public void run() { "
        res = AdvancedCodeAnalyzer.analyze(java_code, "java")
        self.assertGreaterEqual(res["total_errors"], 1)


if __name__ == "__main__":
    unittest.main()
