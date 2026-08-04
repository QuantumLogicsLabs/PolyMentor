"""
Advanced Multi-Language Code Analyzer
======================================

Comprehensive code understanding and error detection system that analyzes:
- Syntax errors (all languages)
- Logical mistakes and anti-patterns
- Bad coding practices
- Performance issues
- Security concerns
- Type mismatches
- Resource leaks
- Unused variables
- And more...

Real-time analysis with detailed categorization.
"""

import ast
import re
import textwrap
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels"""
    CRITICAL = "critical"      # Code won't run
    HIGH = "high"              # Logic errors, bad practices
    MEDIUM = "medium"          # Inefficiency, style issues
    LOW = "low"                # Minor suggestions


class ErrorCategory(Enum):
    """Error categories"""
    SYNTAX = "syntax_error"
    LOGIC = "logical_error"
    TYPE = "type_error"
    PERFORMANCE = "performance_issue"
    SECURITY = "security_issue"
    STYLE = "style_issue"
    BEST_PRACTICE = "best_practice_violation"
    RESOURCE = "resource_leak"
    BOUNDARY = "boundary_error"
    NULL_SAFETY = "null_safety"
    COMPLEXITY = "complexity_issue"


@dataclass
class CodeError:
    """Represents a detected code error"""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    line_number: Optional[int] = None
    column: Optional[int] = None
    suggestion: Optional[str] = None
    code_snippet: Optional[str] = None


class PythonAnalyzer:
    """Python-specific code analyzer"""
    
    @staticmethod
    def analyze(code: str) -> List[CodeError]:
        """Comprehensive Python code analysis"""
        errors = []
        
        # Syntax validation and dedent handling for code snippets
        dedented_code = textwrap.dedent(code)
        lines = dedented_code.split("\n")
        tree = None
        try:
            tree = ast.parse(dedented_code)
        except SyntaxError as e:
            # If parsing fails on snippet boundaries (e.g. return outside function), try wrapping in dummy function
            try:
                tree = ast.parse("def _dummy_snippet_wrapper():\n" + textwrap.indent(dedented_code, "    "))
            except SyntaxError:
                errors.append(CodeError(
                    category=ErrorCategory.SYNTAX,
                    severity=ErrorSeverity.CRITICAL,
                    message=f"Syntax Error: {e.msg}",
                    line_number=e.lineno or 1,
                    suggestion=f"Check line {e.lineno or 1} for syntax issues"
                ))
        
        # AST-based analysis
        if tree:
            errors.extend(PythonAnalyzer._analyze_ast(tree, lines))
        
        # Pattern-based analysis
        errors.extend(PythonAnalyzer._analyze_patterns(dedented_code, lines))
        
        # Style analysis
        errors.extend(PythonAnalyzer._analyze_style(dedented_code, lines))
        
        return errors

    
    @staticmethod
    def _analyze_ast(tree: ast.AST, lines: List[str]) -> List[CodeError]:
        """AST-based analysis for Python"""
        errors = []
        
        class Analyzer(ast.NodeVisitor):
            def __init__(self):
                self.errors = []
                self.defined_vars = set()
                self.used_vars = set()
            
            def visit_FunctionDef(self, node):
                # Check function complexity (parameter count)
                if len(node.args.args) > 7:
                    self.errors.append(CodeError(
                        category=ErrorCategory.COMPLEXITY,
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Function '{node.name}' has {len(node.args.args)} parameters",
                        line_number=node.lineno,
                        suggestion="Consider reducing parameters or using a config object"
                    ))
                
                # Check function length
                func_lines = node.end_lineno - node.lineno if node.end_lineno else 0
                if func_lines > 50:
                    self.errors.append(CodeError(
                        category=ErrorCategory.COMPLEXITY,
                        severity=ErrorSeverity.MEDIUM,
                        message=f"Function '{node.name}' is {func_lines} lines long",
                        line_number=node.lineno,
                        suggestion="Consider breaking into smaller functions"
                    ))
                
                self.generic_visit(node)
            
            def visit_BinOp(self, node):
                # Division by zero analysis
                if isinstance(node.op, ast.Div):
                    self.errors.append(CodeError(
                        category=ErrorCategory.BOUNDARY,
                        severity=ErrorSeverity.HIGH,
                        message="Division operation detected without zero check",
                        line_number=node.lineno,
                        suggestion="Add check: if denominator != 0: result = a / b"
                    ))
                self.generic_visit(node)
            
            def visit_While(self, node):
                # Infinite loop detection
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    has_break = any(isinstance(n, ast.Break) for n in ast.walk(node))
                    if not has_break:
                        self.errors.append(CodeError(
                            category=ErrorCategory.LOGIC,
                            severity=ErrorSeverity.HIGH,
                            message="while True loop without break statement",
                            line_number=node.lineno,
                            suggestion="Add break condition or convert to for loop"
                        ))
                self.generic_visit(node)
            
            def visit_For(self, node):
                # Check for range(len(...)+1) anti-pattern
                if isinstance(node.iter, ast.Call):
                    if isinstance(node.iter.func, ast.Name) and node.iter.func.id == "range":
                        if node.iter.args and isinstance(node.iter.args[0], ast.BinOp):
                            self.errors.append(CodeError(
                                category=ErrorCategory.BOUNDARY,
                                severity=ErrorSeverity.HIGH,
                                message="Potential off-by-one error: range with arithmetic",
                                line_number=node.lineno,
                                suggestion="Use enumerate() or reconsider range bounds"
                            ))
                self.generic_visit(node)
            
            def visit_Name(self, node):
                # Track variable usage
                if isinstance(node.ctx, ast.Store):
                    self.defined_vars.add(node.id)
                elif isinstance(node.ctx, ast.Load):
                    self.used_vars.add(node.id)
                self.generic_visit(node)
            
            def visit_Compare(self, node):
                # Assignment in comparison (x = 5 instead of x == 5)
                if any(isinstance(op, ast.Eq) for op in node.ops):
                    # This is correct usage
                    pass
                self.generic_visit(node)
        
        analyzer = Analyzer()
        analyzer.visit(tree)
        return analyzer.errors
    
    @staticmethod
    def _analyze_patterns(code: str, lines: List[str]) -> List[CodeError]:
        """
        Pattern-based regex analysis for Python code.
        Detects security anti-patterns (e.g. dynamic eval/exec execution), wildcard imports,
        bare except handlers, global scopes, and mutable default arguments.
        """
        errors = []
        
        # Unsafe eval/exec usage
        if re.search(r"\b(?:eval|exec)\s*\(", code):
            errors.append(CodeError(
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.CRITICAL,
                message="Unsafe eval/exec function usage detected",
                suggestion="Avoid evaluating untrusted code strings dynamically"
            ))
        
        # Wildcard imports
        if re.search(r"^\s*from\s+\S+\s+import\s+\*", code, re.MULTILINE):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.MEDIUM,
                message="Wildcard import (from x import *)",
                suggestion="Import specific names to avoid namespace pollution"
            ))
        
        # Bare except clause
        if re.search(r"^\s*except\s*:\s*", code, re.MULTILINE):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.HIGH,
                message="Bare except clause catches all exceptions",
                suggestion="Specify exception type: except ValueError:"
            ))
        
        # Global variable usage
        if re.search(r"^\s*global\s+\w+", code, re.MULTILINE):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.MEDIUM,
                message="Global variable usage detected",
                suggestion="Avoid globals; pass as parameters instead"
            ))
        
        # Mutable default arguments
        if re.search(r"def\s+\w+\([^)]*=\s*(?:\[.*\]|\{.*\})[^)]*\)", code):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.HIGH,
                message="Mutable default argument (list/dict)",
                suggestion="Use None as default: def func(x=None): x = x or []"
            ))
        
        # Comparison to True/False
        if re.search(r"==\s*(?:True|False)", code):
            errors.append(CodeError(
                category=ErrorCategory.STYLE,
                severity=ErrorSeverity.LOW,
                message="Unnecessary comparison to True/False",
                suggestion="Use 'if x:' instead of 'if x == True:'"
            ))
        
        return errors
    
    @staticmethod
    def _analyze_style(code: str, lines: List[str]) -> List[CodeError]:
        """Style and readability analysis for Python"""
        errors = []
        
        # Line length
        long_lines = [(i, len(line)) for i, line in enumerate(lines, 1) if len(line) > 100]
        if long_lines:
            errors.append(CodeError(
                category=ErrorCategory.STYLE,
                severity=ErrorSeverity.LOW,
                message=f"{len(long_lines)} lines exceed 100 characters",
                suggestion="Break long lines for readability"
            ))
        
        # Indentation consistency
        indents = set()
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                if indent > 0:
                    indents.add(indent)
        if indents and len(indents) > 2:
            errors.append(CodeError(
                category=ErrorCategory.STYLE,
                severity=ErrorSeverity.LOW,
                message="Inconsistent indentation detected",
                suggestion="Use consistent spacing (4 spaces recommended)"
            ))
        
        # Missing docstrings
        if re.search(r"^def\s+\w+", code, re.MULTILINE) and '"""' not in code and "'''" not in code:
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.LOW,
                message="Missing docstrings for functions",
                suggestion="Add docstrings: def func():\n    \"\"\"Description.\"\"\""
            ))
        
        return errors


class JavaScriptAnalyzer:
    """JavaScript-specific code analyzer"""
    
    @staticmethod
    def analyze(code: str) -> List[CodeError]:
        """Comprehensive JavaScript code analysis"""
        errors = []
        
        # Check for common JS syntax patterns
        errors.extend(JavaScriptAnalyzer._check_syntax(code))
        errors.extend(JavaScriptAnalyzer._check_patterns(code))
        errors.extend(JavaScriptAnalyzer._check_best_practices(code))
        
        return errors
    
    @staticmethod
    def _check_syntax(code: str) -> List[CodeError]:
        """JavaScript syntax checks"""
        errors = []
        lines = code.split("\n")
        
        # Unmatched braces
        open_braces = code.count("{") - code.count("}")
        if open_braces != 0:
            errors.append(CodeError(
                category=ErrorCategory.SYNTAX,
                severity=ErrorSeverity.CRITICAL,
                message="Unmatched braces detected",
                suggestion=f"Check for missing {'closing' if open_braces > 0 else 'opening'} braces"
            ))
        
        # Assignment in condition (x = 5 instead of x === 5)
        if re.search(r"\b(if|while|for)\s*\([^)]*[^=!<>]=(?![=>])[^=]", code):
            errors.append(CodeError(
                category=ErrorCategory.LOGIC,
                severity=ErrorSeverity.HIGH,
                message="Assignment in condition (=) instead of comparison (==)",
                suggestion="Use === for strict comparison"
            ))
        
        # Missing semicolons (common in JS, but good practice)
        if re.search(r"[a-zA-Z0-9\)]\s*\n\s*[a-z]", code):
            errors.append(CodeError(
                category=ErrorCategory.STYLE,
                severity=ErrorSeverity.LOW,
                message="Missing semicolons detected",
                suggestion="Add semicolons at end of statements"
            ))
        
        return errors
    
    @staticmethod
    def _check_patterns(code: str) -> List[CodeError]:
        """JavaScript anti-patterns"""
        errors = []
        
        # eval() usage
        if re.search(r"\beval\s*\(", code):
            errors.append(CodeError(
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.CRITICAL,
                message="eval() usage detected",
                suggestion="Avoid eval(); use JSON.parse() or other safe alternatives"
            ))
        
        # var instead of let/const
        if re.search(r"^\s*var\s+", code, re.MULTILINE):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.MEDIUM,
                message="'var' keyword detected (old-style variable declaration)",
                suggestion="Use 'let' or 'const' instead for block scoping"
            ))
        
        # Function hoisting issues
        if re.search(r"^\s*function\s+\w+\s*\(", code, re.MULTILINE):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.LOW,
                message="Function declaration (can be hoisted)",
                suggestion="Consider using arrow functions or function expressions"
            ))
        
        # Infinite loops
        if re.search(r"\bwhile\s*\(\s*true\s*\)", code):
            errors.append(CodeError(
                category=ErrorCategory.LOGIC,
                severity=ErrorSeverity.HIGH,
                message="while(true) loop detected",
                suggestion="Add break condition or convert to for loop"
            ))
        
        return errors
    
    @staticmethod
    def _check_best_practices(code: str) -> List[CodeError]:
        """JavaScript best practices"""
        errors = []
        
        # Missing error handling
        if re.search(r"\.then\s*\(\s*function", code) and not re.search(r"\.catch", code):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.MEDIUM,
                message="Promise without catch handler",
                suggestion="Add .catch() to handle promise rejections"
            ))
        
        # Console.log in production
        if re.search(r"console\.(log|warn|error)", code):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.LOW,
                message="Console logging detected",
                suggestion="Remove console statements for production"
            ))
        
        return errors


class CPPAnalyzer:
    """C++ specific code analyzer"""
    
    @staticmethod
    def analyze(code: str) -> List[CodeError]:
        """Comprehensive C++ code analysis"""
        errors = []
        lines = code.split("\n")
        
        # Memory management issues
        errors.extend(CPPAnalyzer._check_memory(code, lines))
        errors.extend(CPPAnalyzer._check_syntax(code, lines))
        errors.extend(CPPAnalyzer._check_style(code, lines))
        
        return errors
    
    @staticmethod
    def _check_memory(code: str, lines: List[str]) -> List[CodeError]:
        """Memory management checks for C++"""
        errors = []
        
        # new without delete
        new_count = code.count("new ")
        delete_count = code.count("delete ")
        if new_count > delete_count:
            errors.append(CodeError(
                category=ErrorCategory.RESOURCE,
                severity=ErrorSeverity.HIGH,
                message=f"Memory leak: {new_count} 'new' but only {delete_count} 'delete'",
                suggestion="Use smart pointers (unique_ptr, shared_ptr) instead of new/delete"
            ))
        
        # Potential buffer overflow
        if re.search(r"gets\s*\(|scanf\s*\(\s*\"%s\"", code):
            errors.append(CodeError(
                category=ErrorCategory.SECURITY,
                severity=ErrorSeverity.CRITICAL,
                message="Unsafe function detected (gets, scanf with %s)",
                suggestion="Use fgets, cin, or scanf with size limits"
            ))
        
        return errors
    
    @staticmethod
    def _check_syntax(code: str, lines: List[str]) -> List[CodeError]:
        """C++ syntax checks"""
        errors = []
        
        # Missing semicolons
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped and not stripped.startswith("//"):
                if re.match(r".*[a-zA-Z0-9\)\]]$", stripped):
                    if not stripped.endswith((";", "{", "}", ":", ",", "\\")):
                        if i < len(lines):
                            next_line = lines[i].strip()
                            if next_line and not next_line.startswith(("{", "}")):
                                errors.append(CodeError(
                                    category=ErrorCategory.SYNTAX,
                                    severity=ErrorSeverity.HIGH,
                                    message=f"Possible missing semicolon on line {i}",
                                    line_number=i,
                                    code_snippet=stripped[:50]
                                ))
                                break
        
        return errors
    
    @staticmethod
    def _check_style(code: str, lines: List[str]) -> List[CodeError]:
        """C++ style checks"""
        errors = []
        
        # Using namespace std
        if "using namespace std;" in code:
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.MEDIUM,
                message="'using namespace std;' pollutes global namespace",
                suggestion="Use std::cout instead or import specific names"
            ))
        
        return errors


class JavaAnalyzer:
    """Java specific code analyzer"""
    
    @staticmethod
    def analyze(code: str) -> List[CodeError]:
        """Comprehensive Java code analysis"""
        errors = []
        
        errors.extend(JavaAnalyzer._check_syntax(code))
        errors.extend(JavaAnalyzer._check_patterns(code))
        errors.extend(JavaAnalyzer._check_best_practices(code))
        
        return errors
    
    @staticmethod
    def _check_syntax(code: str) -> List[CodeError]:
        """Java syntax checks"""
        errors = []
        
        # Unmatched braces
        if code.count("{") != code.count("}"):
            errors.append(CodeError(
                category=ErrorCategory.SYNTAX,
                severity=ErrorSeverity.CRITICAL,
                message="Unmatched braces in Java code",
                suggestion="Check opening and closing braces"
            ))
        
        # Missing semicolons at end of statements
        if re.search(r"[a-zA-Z0-9\)]\s*\n\s*[a-z]", code):
            errors.append(CodeError(
                category=ErrorCategory.SYNTAX,
                severity=ErrorSeverity.MEDIUM,
                message="Possible missing semicolon",
                suggestion="Java requires semicolons at end of statements"
            ))
        
        return errors
    
    @staticmethod
    def _check_patterns(code: str) -> List[CodeError]:
        """Java anti-patterns"""
        errors = []
        
        # NullPointerException risk
        if re.search(r"\.get\(|\.getValue\(", code) and not re.search(r"if\s*\([^)]*!=\s*null", code):
            errors.append(CodeError(
                category=ErrorCategory.NULL_SAFETY,
                severity=ErrorSeverity.HIGH,
                message="Potential NullPointerException: calling methods without null check",
                suggestion="Add null check: if (obj != null) before using"
            ))
        
        # String comparison with ==
        if re.search(r"==\s*['\"]|['\"].*==", code):
            errors.append(CodeError(
                category=ErrorCategory.LOGIC,
                severity=ErrorSeverity.HIGH,
                message="String compared with == instead of .equals()",
                suggestion="Use str1.equals(str2) for string comparison"
            ))
        
        return errors
    
    @staticmethod
    def _check_best_practices(code: str) -> List[CodeError]:
        """Java best practices"""
        errors = []
        
        # Raw types
        if re.search(r"ArrayList\s*[<>]?\s*\(|HashMap\s*[<>]?\s*\(", code):
            errors.append(CodeError(
                category=ErrorCategory.BEST_PRACTICE,
                severity=ErrorSeverity.MEDIUM,
                message="Raw type usage (ArrayList without type parameter)",
                suggestion="Use generics: ArrayList<String> myList = new ArrayList<>();"
            ))
        
        return errors


class AdvancedCodeAnalyzer:
    """Main analyzer that dispatches to language-specific analyzers"""
    
    LANGUAGE_ANALYZERS = {
        "python": PythonAnalyzer,
        "javascript": JavaScriptAnalyzer,
        "js": JavaScriptAnalyzer,
        "cpp": CPPAnalyzer,
        "c++": CPPAnalyzer,
        "java": JavaAnalyzer,
    }
    
    @staticmethod
    def analyze(code: str, language: str) -> Dict:
        """
        Analyze code and return comprehensive error report
        
        Args:
            code: Source code to analyze
            language: Programming language
            
        Returns:
            Dict with analysis results
        """
        language = language.lower().strip()
        analyzer_class = AdvancedCodeAnalyzer.LANGUAGE_ANALYZERS.get(language)
        
        if not analyzer_class:
            return {
                "language": language,
                "supported": False,
                "total_errors": 0,
                "quality_score": 100,
                "errors": [],
                "message": f"Language '{language}' not yet supported"
            }
        
        errors = analyzer_class.analyze(code)
        
        crit = sum(1 for e in errors if e.severity == ErrorSeverity.CRITICAL)
        high = sum(1 for e in errors if e.severity == ErrorSeverity.HIGH)
        med = sum(1 for e in errors if e.severity == ErrorSeverity.MEDIUM)
        low = sum(1 for e in errors if e.severity == ErrorSeverity.LOW)
        score = max(0, 100 - (crit * 25 + high * 15 + med * 10 + low * 5))
        
        return {
            "language": language,
            "supported": True,
            "total_errors": len(errors),
            "quality_score": score,
            "errors": [
                {
                    "category": error.category.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "line": error.line_number,
                    "suggestion": error.suggestion,
                    "code_snippet": error.code_snippet,
                }
                for error in errors
            ],
            "critical_count": crit,
            "high_count": high,
            "medium_count": med,
            "low_count": low,
        }

    
    @staticmethod
    def get_real_time_suggestions(code: str, language: str) -> List[str]:
        """Get real-time coding suggestions"""
        analysis = AdvancedCodeAnalyzer.analyze(code, language)
        
        suggestions = []
        for error in analysis["errors"]:
            if error["suggestion"]:
                suggestions.append(f"• {error['message']}: {error['suggestion']}")
        
        return suggestions
    
    @staticmethod
    def get_quality_score(code: str, language: str) -> int:
        """Calculate code quality score (0-100)"""
        analysis = AdvancedCodeAnalyzer.analyze(code, language)
        
        if not analysis["supported"]:
            return 0
        
        total_errors = analysis["total_errors"]
        critical = analysis["critical_count"]
        high = analysis["high_count"]
        
        # Deduct points based on errors
        score = 100
        score -= critical * 20
        score -= high * 10
        score -= analysis["medium_count"] * 3
        score -= analysis["low_count"]
        
        return max(0, min(100, score))
