from rules import SCORING_RULES, ERROR_EXPLANATIONS, HINT_TEMPLATES


class ExplanationGenerator:
    """Generates clear explanations for detected code issues"""
    
    @staticmethod
    def get_explanation(error_type: str) -> dict:
        """
        Get detailed explanation for a specific error type
        
        Args:
            error_type: The type of error (e.g., 'bad_naming', 'no_comments')
            
        Returns:
            Dictionary with title, explanation, and example
        """
        if error_type in ERROR_EXPLANATIONS:
            return ERROR_EXPLANATIONS[error_type]
        return {"title": "Unknown Error", "explanation": "Unable to provide explanation", "example": ""}


class HintSystem:
    """Provides step-by-step hints to help learners fix issues"""
    
    @staticmethod
    def get_hints(error_type: str) -> list:
        """
        Get step-by-step hints for fixing a specific error
        
        Args:
            error_type: The type of error to fix
            
        Returns:
            List of step-by-step guidance strings
        """
        if error_type in HINT_TEMPLATES:
            return HINT_TEMPLATES[error_type]
        return ["Unable to provide hints for this error type"]


def score_readability(code: str):
    score = SCORING_RULES["readability"]
    issues = []
    issue_details = []

    # Check comments
    if "#" in code:
        score += SCORING_RULES["comments_bonus"]
    else:
        issues.append("No comments found")
        issue_details.append({
            "type": "no_comments",
            "severity": "medium",
            "score_impact": 0,
            "explanation": ExplanationGenerator.get_explanation("no_comments"),
            "hints": HintSystem.get_hints("no_comments")
        })

    # Poor variable naming (single letters)
    words = code.split()
    if any(len(word) == 1 for word in words):
        score += SCORING_RULES["bad_naming_penalty"]
        issues.append("Poor variable naming detected")
        issue_details.append({
            "type": "bad_naming",
            "severity": "low",
            "score_impact": SCORING_RULES["bad_naming_penalty"],
            "explanation": ExplanationGenerator.get_explanation("bad_naming"),
            "hints": HintSystem.get_hints("bad_naming")
        })

    return max(score, 0), issues, issue_details


def score_logic(code: str):
    score = SCORING_RULES["logic"]
    issues = []
    issue_details = []

    # Too many loops
    if code.count("for") > 3:
        score += SCORING_RULES["loop_penalty"]
        issues.append("Too many loops")
        issue_details.append({
            "type": "too_many_loops",
            "severity": "medium",
            "score_impact": SCORING_RULES["loop_penalty"],
            "explanation": ExplanationGenerator.get_explanation("too_many_loops"),
            "hints": HintSystem.get_hints("too_many_loops")
        })

    # Infinite loop risk
    if "while True" in code:
        score -= 10
        issues.append("Potential infinite loop")
        issue_details.append({
            "type": "infinite_loop",
            "severity": "high",
            "score_impact": -10,
            "explanation": ExplanationGenerator.get_explanation("infinite_loop"),
            "hints": HintSystem.get_hints("infinite_loop")
        })

    return max(score, 0), issues, issue_details


def score_efficiency(code: str):
    score = SCORING_RULES["efficiency"]
    issues = []
    issue_details = []

    # Nested loops (basic detection)
    if code.count("for") > 1:
        score += SCORING_RULES["deep_nesting_penalty"]
        issues.append("Possible nested loops (O(n^2))")
        issue_details.append({
            "type": "deep_nesting",
            "severity": "medium",
            "score_impact": SCORING_RULES["deep_nesting_penalty"],
            "explanation": ExplanationGenerator.get_explanation("deep_nesting"),
            "hints": HintSystem.get_hints("deep_nesting")
        })

    return max(score, 0), issues, issue_details


def evaluate_code(code: str):
    """
    Comprehensive code evaluation with scoring, issues, explanations, and hints
    
    Args:
        code: Python code string to evaluate
        
    Returns:
        Dictionary with total score, breakdown, issues, and learning materials
    """
    r_score, r_issues, r_details = score_readability(code)
    l_score, l_issues, l_details = score_logic(code)
    e_score, e_issues, e_details = score_efficiency(code)

    total = r_score + l_score + e_score
    all_issue_details = r_details + l_details + e_details

    return {
        "total_score": max(total, 0),
        "breakdown": {
            "readability": r_score,
            "logic": l_score,
            "efficiency": e_score
        },
        "issues": r_issues + l_issues + e_issues,
        "detailed_issues": all_issue_details,
        "summary": generate_feedback_summary(total, all_issue_details)
    }


def generate_feedback_summary(score: int, issues: list) -> str:
    """
    Generate human-friendly feedback summary
    
    Args:
        score: Total code score
        issues: List of detected issues
        
    Returns:
        Formatted feedback string
    """
    if not issues:
        return "Excellent code! No major issues detected. Keep up the great work!"
    
    feedback = f"Score: {max(score, 0)}/80 - "
    
    if score >= 60:
        feedback += "Good work! A few improvements:"
    elif score >= 40:
        feedback += "There are some areas to improve:"
    else:
        feedback += "Let's focus on these issues to improve your code:"
    
    return feedback