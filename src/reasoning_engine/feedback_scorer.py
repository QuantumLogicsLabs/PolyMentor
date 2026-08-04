import ast
import re
from typing import Dict, List


class FeedbackScorer:
    """
    Scores code quality and tracks hint feedback for adaptive learning.
    
    Features:
    - Code quality scoring (0-100)
    - Hint effectiveness tracking
    - Learner performance analytics
    - Adaptive difficulty recommendations
    """
    
    def __init__(self):
        """Initialize feedback tracker"""
        self.hint_feedback: Dict[str, List[Dict]] = {}
        self.learner_performance: Dict[str, Dict] = {}

    def score(self, code: str, language: str = "python") -> int:
        """Score code quality on 0-100 scale"""
        score = 100

        lines = code.strip().split("\n")

        # Penalize very long lines
        long_lines = sum(1 for l in lines if len(l) > 100)
        score -= long_lines * 3

        # Penalize deeply nested code (4+ levels of indentation)
        deep_nesting = sum(1 for l in lines if l.startswith("    " * 4))
        score -= deep_nesting * 5

        # Penalize magic numbers
        magic_numbers = len(re.findall(r"\b(?<!\.)\d{2,}\b", code))
        score -= magic_numbers * 2

        # Penalize very short variable names (single char, excluding i/j/k/n/x/y)
        bad_vars = len(re.findall(r"\b(?![ijknxy])[a-wz]\b\s*=", code))
        score -= bad_vars * 3

        # Penalize no comments/docstrings in longer functions
        if len(lines) > 15 and '"""' not in code and "#" not in code:
            score -= 10

        return max(0, min(100, score))

    def score_feedback(
        self,
        hint_text: str,
        was_helpful: bool,
        error_type: str,
        user_level: str
    ) -> int:
        """
        Score hint feedback and update performance metrics.
        
        Args:
            hint_text: The hint that was shown
            was_helpful: Whether the learner found it helpful
            error_type: Type of error (e.g., 'syntax_error', 'loop_boundaries')
            user_level: Learner level (beginner, intermediate, advanced)
        
        Returns:
            Score 0-100 indicating how well the hint worked
        """
        # Calculate feedback score
        base_score = 100 if was_helpful else 50
        
        # Adjust based on hint length (too long or too short both penalized)
        hint_len = len(hint_text.split())
        if hint_len < 5:  # Too short, not helpful
            base_score -= 25
        elif hint_len > 100:  # Too long, overwhelming
            base_score -= 15
        
        # Track feedback for this error type
        if error_type not in self.hint_feedback:
            self.hint_feedback[error_type] = []
        
        feedback_record = {
            "hint_text": hint_text,
            "was_helpful": was_helpful,
            "score": base_score,
            "user_level": user_level,
            "hint_length": hint_len
        }
        self.hint_feedback[error_type].append(feedback_record)
        
        return base_score

    def get_hint_effectiveness(self, error_type: str) -> Dict:
        """
        Get effectiveness statistics for hints of an error type.
        
        Returns:
            Dict with success rate, avg score, and recommendation
        """
        if error_type not in self.hint_feedback:
            return {"status": "no_data"}
        
        feedbacks = self.hint_feedback[error_type]
        helpful_count = sum(1 for f in feedbacks if f["was_helpful"])
        total_count = len(feedbacks)
        avg_score = sum(f["score"] for f in feedbacks) / total_count
        
        return {
            "error_type": error_type,
            "total_feedback_records": total_count,
            "helpful_count": helpful_count,
            "success_rate": round(helpful_count / total_count, 2) if total_count > 0 else 0,
            "average_score": round(avg_score, 1),
            "recommendation": self._get_hint_recommendation(helpful_count, total_count)
        }

    def _get_hint_recommendation(self, helpful_count: int, total_count: int) -> str:
        """Get recommendation based on success rate"""
        if total_count < 3:
            return "Need more data"
        
        success_rate = helpful_count / total_count
        if success_rate >= 0.8:
            return "Excellent - keep using this hint"
        elif success_rate >= 0.5:
            return "Good - but consider refining"
        else:
            return "Needs improvement - consider revising"

    def recommend_difficulty_change(
        self,
        user_level: str,
        success_rate: float,
        errors_solved: int
    ) -> str:
        """
        Recommend difficulty change based on performance.
        
        Args:
            user_level: Current difficulty level
            success_rate: Percentage of problems solved (0-1)
            errors_solved: Number of errors successfully resolved
        
        Returns:
            Recommended difficulty level
        """
        if success_rate >= 0.85 and errors_solved >= 5:
            if user_level == "beginner":
                return "intermediate"
            elif user_level == "intermediate":
                return "advanced"
        elif success_rate < 0.5 and user_level != "beginner":
            if user_level == "advanced":
                return "intermediate"
            elif user_level == "intermediate":
                return "beginner"
        
        return user_level

    def track_learner_session(
        self,
        learner_id: str,
        error_type: str,
        level: str,
        hints_used: int,
        problem_solved: bool
    ) -> None:
        """
        Track a learning session for analytics.
        
        Args:
            learner_id: Unique learner identifier
            error_type: Type of error encountered
            level: Difficulty level
            hints_used: Number of hints used
            problem_solved: Whether problem was solved
        """
        if learner_id not in self.learner_performance:
            self.learner_performance[learner_id] = {
                "total_sessions": 0,
                "total_errors": 0,
                "solved_count": 0,
                "error_types": {}
            }
        
        perf = self.learner_performance[learner_id]
        perf["total_sessions"] += 1
        perf["total_errors"] += 1
        if problem_solved:
            perf["solved_count"] += 1
        
        if error_type not in perf["error_types"]:
            perf["error_types"][error_type] = {
                "encountered": 0,
                "solved": 0,
                "total_hints_used": 0
            }
        
        perf["error_types"][error_type]["encountered"] += 1
        if problem_solved:
            perf["error_types"][error_type]["solved"] += 1
        perf["error_types"][error_type]["total_hints_used"] += hints_used

    def get_learner_analytics(self, learner_id: str) -> Dict:
        """Get analytics for a specific learner"""
        if learner_id not in self.learner_performance:
            return {"status": "no_data"}
        
        perf = self.learner_performance[learner_id]
        solved_rate = perf["solved_count"] / perf["total_sessions"] if perf["total_sessions"] > 0 else 0
        
        return {
            "learner_id": learner_id,
            "total_sessions": perf["total_sessions"],
            "problems_solved": perf["solved_count"],
            "success_rate": round(solved_rate, 2),
            "error_types_encountered": list(perf["error_types"].keys()),
            "most_common_error": max(perf["error_types"].items(), key=lambda x: x[1]["encountered"])[0] if perf["error_types"] else None,
            "performance_data": perf["error_types"]
        }
