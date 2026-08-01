"""
src/inference/context_builder.py
--------------------------------
Builds execution and error context for the PolyMentor pipeline.
Integrates AdvancedCodeAnalyzer to provide the model with deep
insights about the code before it responds.
"""

from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer

class ContextBuilder:
    """Builds structured context for the inference pipeline."""
    
    @staticmethod
    def build_analyzer_context(code: str, language: str) -> str:
        """
        Runs the AdvancedCodeAnalyzer on the provided code and formats
        the results into a markdown context string for the prompt.
        """
        if not code or not code.strip():
            return ""

        analysis = AdvancedCodeAnalyzer.analyze(code, language)
        
        if not analysis.get("supported", False):
            return ""

        total_errors = analysis.get("total_errors", 0)
        
        if total_errors == 0:
            return "Analyzer Status: Clean (No syntax or logical errors detected by static analysis)."
        
        context_lines = [
            f"Analyzer Status: Found {total_errors} issue(s).",
            "Details:"
        ]
        
        for idx, error in enumerate(analysis.get("errors", []), 1):
            severity = error.get('severity', 'unknown').upper()
            category = error.get('category', 'unknown')
            msg = error.get('message', '')
            line = error.get('line')
            
            line_info = f" at line {line}" if line else ""
            context_lines.append(f"  {idx}. [{severity}] ({category}){line_info}: {msg}")
            
            suggestion = error.get('suggestion')
            if suggestion:
                context_lines.append(f"     Suggestion: {suggestion}")

        quality_score = AdvancedCodeAnalyzer.get_quality_score(code, language)
        context_lines.append(f"\nStatic Quality Score: {quality_score}/100")
        
        return "\n".join(context_lines)
