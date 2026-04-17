"""
Test cases for ErrorDetector module
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.error_detector import ErrorDetector


def test_python_syntax_error():
    """Test Python syntax error detection."""
    detector = ErrorDetector()
    code = "def hello()\n    print('Hello World')"
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == True
    assert result["error_type"] == "syntax_error"
    assert result["subtype"] == "python_syntax_error"
    assert result["line"] == 1
    assert result["language"] == "python"
    
    print("Test 1 PASSED: Python syntax error detected")


def test_missing_colon():
    """Test missing colon detection in Python."""
    detector = ErrorDetector()
    code = "if x > 5\n    print('x is greater than 5')"
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == True
    assert result["error_type"] == "syntax_error"
    assert result["subtype"] == "python_syntax_error"  # ast.parse catches this first
    assert result["line"] == 1
    assert "expected ':'" in result["message"]  # Python's error message
    
    print("Test 2 PASSED: Missing colon detected")


def test_assignment_in_condition():
    """Test assignment instead of comparison detection."""
    detector = ErrorDetector()
    code = "if x = 5:\n    print('x is 5')"
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == True
    assert result["error_type"] == "syntax_error"
    assert result["subtype"] == "python_syntax_error"  # ast.parse catches this first
    assert result["line"] == 1
    assert "invalid syntax" in result["message"]  # Python's error message
    
    print("Test 3 PASSED: Assignment in condition detected")


def test_infinite_loop():
    """Test infinite loop detection."""
    detector = ErrorDetector()
    code = """while True:
    print('This will run forever')
    print('Still running')
"""
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == True
    assert result["error_type"] == "logical_error"
    assert result["subtype"] == "possible_infinite_loop"
    assert result["line"] == 1
    assert "infinite loop" in result["message"]
    
    print("Test 4 PASSED: Infinite loop detected")


def test_correct_code():
    """Test correct code detection (no errors)."""
    detector = ErrorDetector()
    code = """def fibonacci(n):
    if n <= 1:
        return n
    else:
        return fibonacci(n-1) + fibonacci(n-2)

result = fibonacci(10)
print(f"Fibonacci of 10 is {result}")
"""
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == False
    assert result["error_type"] == None
    assert result["subtype"] == None
    assert result["line"] == None
    assert result["message"] == "No obvious error detected"
    assert result["language"] == "python"
    
    print("Test 5 PASSED: Correct code detected as error-free")


def test_unmatched_brackets():
    """Test unmatched brackets detection."""
    detector = ErrorDetector()
    code = "def hello():\n    print('Hello World'"
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == True
    assert result["error_type"] == "syntax_error"
    assert result["subtype"] == "python_syntax_error"  # ast.parse catches this first
    assert result["line"] == 2
    assert "never closed" in result["message"]  # Python's error message
    
    print("Test 6 PASSED: Unmatched brackets detected")


def test_divide_by_zero_risk():
    """Test divide by zero risk detection."""
    detector = ErrorDetector()
    code = "result = 100 / x"
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == True
    assert result["error_type"] == "logical_error"
    assert result["subtype"] == "divide_by_zero_risk"
    assert result["line"] == 1
    assert "divide by zero" in result["message"]
    
    print("Test 7 PASSED: Divide by zero risk detected")


def test_off_by_one_error():
    """Test off-by-one error detection."""
    detector = ErrorDetector()
    code = """arr = [1, 2, 3, 4, 5]
for i in range(len(arr)):
    print(arr[i+1])
"""
    
    result = detector.detect(code, "python")
    
    assert result["has_error"] == True
    assert result["error_type"] == "logical_error"
    assert result["subtype"] == "off_by_one"
    assert result["line"] == 3
    assert "off-by-one" in result["message"]
    
    print("Test 8 PASSED: Off-by-one error detected")


def test_javascript_support():
    """Test JavaScript language support."""
    detector = ErrorDetector()
    code = "function hello() {\n    console.log('Hello World'"
    
    result = detector.detect(code, "javascript")
    
    assert result["has_error"] == True
    assert result["error_type"] == "syntax_error"
    assert result["subtype"] == "unmatched_brackets"
    assert result["language"] == "javascript"
    
    print("Test 9 PASSED: JavaScript error detection works")


def test_unsupported_language():
    """Test unsupported language handling."""
    detector = ErrorDetector()
    code = "print('Hello')"
    
    result = detector.detect(code, "ruby")
    
    assert result["has_error"] == False
    assert result["error_type"] == None
    assert "Unsupported language" in result["message"]
    assert result["language"] == "ruby"
    
    print("Test 10 PASSED: Unsupported language handled correctly")


if __name__ == "__main__":
    print("Running ErrorDetector tests...")
    print("=" * 50)
    
    try:
        test_python_syntax_error()
        test_missing_colon()
        test_assignment_in_condition()
        test_infinite_loop()
        test_correct_code()
        test_unmatched_brackets()
        test_divide_by_zero_risk()
        test_off_by_one_error()
        test_javascript_support()
        test_unsupported_language()
        
        print("=" * 50)
        print("All tests PASSED! ErrorDetector is working correctly.")
        
    except Exception as e:
        print(f"Test FAILED: {e}")
        import traceback
        traceback.print_exc()
