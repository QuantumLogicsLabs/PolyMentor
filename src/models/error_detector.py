"""
Error Detection Module for PolyMentor AI Coding Assistant

A rule-based error detection system that analyzes code in multiple languages
and detects syntax errors and common logical errors.
"""

import ast
import re
from typing import Dict, Optional, List


class ErrorDetector:
    """
    Rule-based error detector for multiple programming languages.
    
    Supports: Python, JavaScript, Java, C++
    Detects: Syntax errors, Common logical errors
    """
    
    def detect(self, code: str, language: str) -> Dict:
        """
        Main method to detect errors in code.
        
        Args:
            code: Source code as string
            language: Programming language (python, javascript, java, cpp)
            
        Returns:
            Dict containing error information with format:
            {
                "has_error": bool,
                "error_type": "syntax_error" or "logical_error" or None,
                "subtype": str or None,
                "line": int or None,
                "message": str,
                "language": str
            }
        """
        language = language.lower()
        
        # Route to appropriate language detector
        if language == "python":
            return self._detect_python(code, language)
        elif language == "javascript":
            return self._detect_javascript(code, language)
        elif language == "java":
            return self._detect_java(code, language)
        elif language == "cpp":
            return self._detect_cpp(code, language)
        else:
            return {
                "has_error": False,
                "error_type": None,
                "subtype": None,
                "line": None,
                "message": f"Unsupported language: {language}",
                "language": language
            }
    
    def _detect_python(self, code: str, language: str) -> Dict:
        """Detect errors in Python code."""
        # First check Python syntax using ast module
        try:
            ast.parse(code)
        except SyntaxError as e:
            return {
                "has_error": True,
                "error_type": "syntax_error",
                "subtype": "python_syntax_error",
                "line": e.lineno,
                "message": e.msg,
                "language": language
            }
        
        # Check generic syntax rules
        syntax_error = self._check_generic_syntax_rules(code, language)
        if syntax_error:
            return syntax_error
        
        # Check logical errors
        logical_error = self._check_logical_errors(code, language)
        if logical_error:
            return logical_error
        
        # No error found
        return self._no_error_result(language)
    
    def _detect_javascript(self, code: str, language: str) -> Dict:
        """Detect errors in JavaScript code."""
        # Check generic syntax rules
        syntax_error = self._check_generic_syntax_rules(code, language)
        if syntax_error:
            return syntax_error
        
        # Check logical errors
        logical_error = self._check_logical_errors(code, language)
        if logical_error:
            return logical_error
        
        return self._no_error_result(language)
    
    def _detect_java(self, code: str, language: str) -> Dict:
        """Detect errors in Java code."""
        # Check generic syntax rules
        syntax_error = self._check_generic_syntax_rules(code, language)
        if syntax_error:
            return syntax_error
        
        # Check logical errors
        logical_error = self._check_logical_errors(code, language)
        if logical_error:
            return logical_error
        
        return self._no_error_result(language)
    
    def _detect_cpp(self, code: str, language: str) -> Dict:
        """Detect errors in C++ code."""
        # Check generic syntax rules
        syntax_error = self._check_generic_syntax_rules(code, language)
        if syntax_error:
            return syntax_error
        
        # Check logical errors
        logical_error = self._check_logical_errors(code, language)
        if logical_error:
            return logical_error
        
        return self._no_error_result(language)
    
    def _check_generic_syntax_rules(self, code: str, language: str) -> Optional[Dict]:
        """Check generic syntax rules applicable to all languages."""
        lines = code.split('\n')
        
        # Check for unmatched brackets
        bracket_error = self._check_brackets(code, language)
        if bracket_error:
            return bracket_error
        
        # Check for missing colon (Python specific)
        if language == "python":
            colon_error = self._check_missing_colon(lines, language)
            if colon_error:
                return colon_error
        
        return None
    
    def _check_brackets(self, code: str, language: str) -> Optional[Dict]:
        """Check for unmatched brackets: (), {}, []"""
        stack = []
        bracket_pairs = {')': '(', '}': '{', ']': '['}
        opening_brackets = set(bracket_pairs.values())
        
        lines = code.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for char_num, char in enumerate(line):
                if char in opening_brackets:
                    stack.append((char, line_num, char_num))
                elif char in bracket_pairs:
                    if not stack or stack[-1][0] != bracket_pairs[char]:
                        return {
                            "has_error": True,
                            "error_type": "syntax_error",
                            "subtype": "unmatched_brackets",
                            "line": line_num,
                            "message": f"Unmatched '{char}' at position {char_num}",
                            "language": language
                        }
                    stack.pop()
        
        if stack:
            unclosed_bracket, line_num, char_num = stack[-1]
            return {
                "has_error": True,
                "error_type": "syntax_error",
                "subtype": "unmatched_brackets",
                "line": line_num,
                "message": f"Unclosed '{unclosed_bracket}' at position {char_num}",
                "language": language
            }
        
        return None
    
    def _check_missing_colon(self, lines: List[str], language: str) -> Optional[Dict]:
        """Check for missing colon in Python control structures."""
        colon_keywords = {'if', 'for', 'while', 'def', 'class', 'elif', 'else', 'try', 'except', 'finally', 'with'}
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Check if line starts with a keyword that should have a colon
            words = stripped.split()
            if words and words[0] in colon_keywords:
                # All these keywords should end with a colon
                if ':' not in line:
                    return {
                        "has_error": True,
                        "error_type": "syntax_error",
                        "subtype": "missing_colon",
                        "line": line_num,
                        "message": f"Missing colon after '{words[0]}' statement",
                        "language": language
                    }
        
        return None
    
    def _check_logical_errors(self, code: str, language: str) -> Optional[Dict]:
        """Check for common logical errors."""
        lines = code.split('\n')
        
        # Check for assignment instead of comparison
        assignment_error = self._check_assignment_in_condition(lines, language)
        if assignment_error:
            return assignment_error
        
        # Check for off-by-one errors
        off_by_one_error = self._check_off_by_one_errors(lines, language)
        if off_by_one_error:
            return off_by_one_error
        
        # Check for possible infinite loops
        infinite_loop_error = self._check_infinite_loops(lines, language)
        if infinite_loop_error:
            return infinite_loop_error
        
        # Check for divide by zero risk
        div_zero_error = self._check_divide_by_zero_risk(lines, language)
        if div_zero_error:
            return div_zero_error
        
        return None
    
    def _check_assignment_in_condition(self, lines: List[str], language: str) -> Optional[Dict]:
        """Check for assignment instead of comparison in conditions."""
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Pattern: if x = 5 (assignment in condition)
            # Use negative lookbehind and lookahead to avoid matching comparison operators
            if_match = re.search(r'if\s*\([^<>!=]*=[^=]*\)', stripped)
            if not if_match:
                if_match = re.search(r'if\s+[^<>!=]*=[^=]*:', stripped)
            
            if if_match:
                return {
                    "has_error": True,
                    "error_type": "logical_error",
                    "subtype": "assignment_in_condition",
                    "line": line_num,
                    "message": "Possible assignment instead of comparison in condition",
                    "language": language
                }
        
        return None
    
    def _check_off_by_one_errors(self, lines: List[str], language: str) -> Optional[Dict]:
        """Check for potential off-by-one errors."""
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Pattern: for i in range(len(arr)): followed by arr[i+1]
            if 'range(len(' in stripped and ')' in stripped:
                # Look ahead a few lines for arr[i+1] pattern
                for future_line_num in range(line_num, min(line_num + 5, len(lines))):
                    future_line = lines[future_line_num].strip()
                    if '[i+1]' in future_line or '[i + 1]' in future_line:
                        return {
                            "has_error": True,
                            "error_type": "logical_error",
                            "subtype": "off_by_one",
                            "line": future_line_num + 1,
                            "message": "Potential off-by-one error: accessing array element beyond bounds",
                            "language": language
                        }
            
            # Pattern: <= len(arr) in loop condition
            if '<= len(' in stripped:
                return {
                    "has_error": True,
                    "error_type": "logical_error",
                    "subtype": "off_by_one",
                    "line": line_num,
                    "message": "Potential off-by-one error: using <= with array length",
                    "language": language
                }
        
        return None
    
    def _check_infinite_loops(self, lines: List[str], language: str) -> Optional[Dict]:
        """Check for possible infinite loops."""
        in_loop = False
        loop_start_line = 0
        has_break = False
        loop_indent_level = 0
        
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Detect loop start
            if 'while True' in stripped or 'while(1)' in stripped or 'while (1)' in stripped:
                in_loop = True
                loop_start_line = line_num
                has_break = False
                loop_indent_level = len(line) - len(line.lstrip())
                continue
            
            # Check for break statement inside loop
            if in_loop and 'break' in stripped:
                current_indent = len(line) - len(line.lstrip())
                # Only count break if it's at the same or deeper indent level as the loop
                if current_indent >= loop_indent_level:
                    has_break = True
                    in_loop = False
            
            # Check if loop ended (dedentation to same or lower level than loop start)
            if in_loop and line_num > loop_start_line:
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= loop_indent_level and stripped:
                    # Loop ended without break
                    if not has_break:
                        return {
                            "has_error": True,
                            "error_type": "logical_error",
                            "subtype": "possible_infinite_loop",
                            "line": loop_start_line,
                            "message": "Possible infinite loop: while True without break statement",
                            "language": language
                        }
                    in_loop = False
        
        # If we're still in loop at end of file and no break found
        if in_loop and not has_break:
            return {
                "has_error": True,
                "error_type": "logical_error",
                "subtype": "possible_infinite_loop",
                "line": loop_start_line,
                "message": "Possible infinite loop: while True without break statement",
                "language": language
            }
        
        return None
    
    def _check_divide_by_zero_risk(self, lines: List[str], language: str) -> Optional[Dict]:
        """Check for potential divide by zero errors."""
        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith('#'):
                continue
            
            # Pattern: / variable where variable could be zero
            div_match = re.search(r'/\s*([a-zA-Z_][a-zA-Z0-9_]*)', stripped)
            if div_match:
                variable = div_match.group(1)
                # Look for zero check in surrounding lines
                has_zero_check = False
                
                # Check previous 3 lines and next 3 lines
                start_check = max(0, line_num - 4)
                end_check = min(len(lines), line_num + 3)
                
                for check_line_num in range(start_check, end_check):
                    check_line = lines[check_line_num].strip()
                    if f'{variable} == 0' in check_line or f'{variable} != 0' in check_line:
                        has_zero_check = True
                        break
                
                if not has_zero_check:
                    return {
                        "has_error": True,
                        "error_type": "logical_error",
                        "subtype": "divide_by_zero_risk",
                        "line": line_num,
                        "message": f"Potential divide by zero: variable '{variable}' not checked for zero",
                        "language": language
                    }
        
        return None
    
    def _no_error_result(self, language: str) -> Dict:
        """Return default result when no error is found."""
        return {
            "has_error": False,
            "error_type": None,
            "subtype": None,
            "line": None,
            "message": "No obvious error detected",
            "language": language
        }
