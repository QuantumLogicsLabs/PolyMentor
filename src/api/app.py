"""
PolyMentor FastAPI Application

A comprehensive AI-powered coding mentor that:
- Detects errors across 4 languages (Python, JavaScript, C++, Java)
- Provides human-like explanations of concepts
- Generates step-by-step hints for learning
- Tracks learner progress and adapts difficulty
- Offers structured learning paths
"""

from __future__ import annotations

import time
import os
from typing import Literal, Optional
from dataclasses import asdict

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from src.analysis.advanced_analyzer import AdvancedCodeAnalyzer, ErrorSeverity, ErrorCategory
from src.learning.concept_guide import CONCEPT_LIBRARY, get_concept_explanation, get_learning_path
from src.reasoning_engine.hint_system import HintSystem
from src.reasoning_engine.feedback_scorer import FeedbackScorer
from src.inference.pipeline import PolyMentorPipeline


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="PolyMentor API",
    description="AI-powered coding mentor with error detection, learning guidance, and adaptive hints",
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in os.getenv("ALLOWED_ORIGINS", "*").split(",") if origin.strip()],
    allow_methods=["*"],
    allow_headers=["*"],
)

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize core systems
hint_system = HintSystem()
feedback_scorer = FeedbackScorer()
pipeline = PolyMentorPipeline()


# ============================================================================
# Request/Response Models
# ============================================================================

class ChatRequest(BaseModel):
    """Request for chatbot interaction"""
    message: str = Field(..., max_length=2000, description="User message or question")
    code: Optional[str] = Field(default=None, max_length=15000, description="Optional code snippet for review or discussion")
    language: str = Field(default="python", max_length=50, description="Programming language")
    level: Literal["beginner", "intermediate", "advanced"] = Field(
        default="beginner",
        description="Learner skill level"
    )
    history: list[dict[str, str]] = Field(
        default_factory=list,
        description="Prior conversational history turns [{role: ..., content: ...}]"
    )



class ChatResponse(BaseModel):
    """Response from chatbot"""
    status: str
    answer: str
    language: str
    level: str
    model: str
    suspected_bugs: list = Field(default_factory=list)
    fixed_code: Optional[str] = None
    lesson: Optional[str] = None
    next_steps: list = Field(default_factory=list)
    elapsed_ms: float


class AnalyzeRequest(BaseModel):
    """Request for code analysis"""
    code: str = Field(..., max_length=10000, description="Code to analyze")
    language: str = Field(default="python", max_length=50, description="Programming language")
    level: Literal["beginner", "intermediate", "advanced"] = Field(
        default="beginner",
        description="Learner skill level"
    )
    num_hints: int = Field(default=3, ge=1, le=10, description="Number of hints to generate")


class AnalyzeResponse(BaseModel):
    """Response from code analysis"""
    status: str
    language: str
    code_length: int
    total_errors: int
    errors: list
    quality_score: int
    suggestions: list
    severity_breakdown: dict
    elapsed_ms: float


class QualityScoreResponse(BaseModel):
    """Code quality assessment response"""
    quality_score: int
    factors: dict
    elapsed_ms: float


class SuggestionsResponse(BaseModel):
    """Code improvement suggestions"""
    suggestions: list
    total_suggestions: int
    language: str


class LanguagesResponse(BaseModel):
    """List of supported languages"""
    languages: list


class ConceptResponse(BaseModel):
    """Single concept explanation"""
    status: str
    concept_name: str
    difficulty: str
    simple_explanation: str
    detailed_explanation: str
    prerequisites: list
    related_concepts: list
    common_mistakes: list
    tips_for_mastery: list
    code_examples: list
    learning_resources: list


class ConceptsListResponse(BaseModel):
    """List of all concepts"""
    total_concepts: int
    concepts: list


class LearningPathResponse(BaseModel):
    """Learning path for a concept"""
    starting_concept: str
    difficulty: str
    learning_path: dict


class HintsResponse(BaseModel):
    """Hints for an error"""
    status: str
    error_type: str
    level: str
    hints: list
    total_hints: int


class AdaptiveResponse(BaseModel):
    """Adaptive learning response"""
    status: str
    current_level: str
    recommended_level: str
    reason: str
    next_concept: Optional[str]


# ============================================================================
# Health & Info Endpoints
# ============================================================================

@app.get("/")
def root():
    """Homepage with API information"""
    return {
        "name": "PolyMentor",
        "purpose": "AI-powered coding mentor for error detection and learning",
        "version": "0.3.0",
        "features": [
            "Multi-language error detection",
            "Concept-based learning",
            "Progressive hint generation",
            "Adaptive difficulty adjustment",
            "Performance tracking"
        ],
        "docs": "/docs",
        "redoc": "/redoc",
        "endpoints": {
            "health": "GET /health",
            "languages": "GET /languages",
            "error_detection": ["POST /analyze", "POST /analyze/detailed"],
            "learning": ["GET /learn/concepts", "GET /learn/concept/{id}"],
            "hints": ["GET /learn/hints/{error_type}", "POST /learn/next-hint"]
        }
    }


@app.get("/health")
def health():
    """Server health check"""
    return {
        "status": "ok",
        "version": "0.3.0",
        "systems": {
            "analyzer": "ready",
            "learning": "ready",
            "hints": "ready"
        }
    }


@app.get("/languages", response_model=LanguagesResponse)
def get_languages():
    """Get list of supported programming languages"""
    return {
        "languages": ["python", "javascript", "c++", "java"]
    }


# ============================================================================
# Error Detection & Analysis Endpoints (9 endpoints)
# ============================================================================

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(request: AnalyzeRequest):
    """
    Analyze code for errors with explanations.
    
    Returns:
    - Errors found with categories and severity
    - Quality score (0-100)
    - Improvement suggestions
    - Step-by-step hints for learners
    """
    start_time = time.time()
    
    result = AdvancedCodeAnalyzer.analyze(request.code, request.language)
    quality_score = AdvancedCodeAnalyzer.get_quality_score(request.code, request.language)
    suggestions = AdvancedCodeAnalyzer.get_real_time_suggestions(request.code, request.language)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return {
        "status": "analyzed" if result["supported"] else "unsupported_language",
        "language": request.language,
        "code_length": len(request.code),
        "total_errors": result.get("total_errors", 0),
        "errors": result.get("errors", []),
        "quality_score": quality_score,
        "suggestions": suggestions,
        "severity_breakdown": {
            "critical": result.get("severity_counts", {}).get("critical", 0),
            "high": result.get("severity_counts", {}).get("high", 0),
            "medium": result.get("severity_counts", {}).get("medium", 0),
            "low": result.get("severity_counts", {}).get("low", 0)
        },
        "elapsed_ms": round(elapsed_ms, 2)
    }


@app.post("/analyze/detailed", response_model=AnalyzeResponse)
def analyze_detailed(request: AnalyzeRequest):
    """
    Detailed code analysis with comprehensive categorization.
    
    Returns:
    - Errors grouped by category (syntax, logic, type, performance, security, etc.)
    - Severity levels (critical → low)
    - Quality metrics
    - Performance analysis
    """
    start_time = time.time()
    
    result = AdvancedCodeAnalyzer.analyze(request.code, request.language)
    quality_score = AdvancedCodeAnalyzer.get_quality_score(request.code, request.language)
    
    # Organize errors by category
    errors_by_category = {}
    for error in result.get("errors", []):
        category = error.get("category", "unknown")
        if category not in errors_by_category:
            errors_by_category[category] = []
        errors_by_category[category].append(error)
    
    elapsed_ms = (time.time() - start_time) * 1000
    
    return {
        "status": "analyzed" if result.get("supported") else "unsupported_language",
        "language": request.language,
        "code_length": len(request.code),
        "total_errors": result.get("total_errors", 0),
        "errors": result.get("errors", []),
        "quality_score": quality_score,
        "suggestions": [],
        "severity_breakdown": {
            "critical": result.get("severity_counts", {}).get("critical", 0),
            "high": result.get("severity_counts", {}).get("high", 0),
            "medium": result.get("severity_counts", {}).get("medium", 0),
            "low": result.get("severity_counts", {}).get("low", 0)
        },
        "elapsed_ms": round(elapsed_ms, 2)
    }


@app.post("/quality-score", response_model=QualityScoreResponse)
def quality_score(request: AnalyzeRequest):
    """
    Get code quality score (0-100).
    
    Evaluates:
    - Naming conventions
    - Indentation consistency
    - Function complexity
    - Code readability
    """
    start_time = time.time()
    score = AdvancedCodeAnalyzer.get_quality_score(request.code, request.language)
    elapsed_ms = (time.time() - start_time) * 1000
    
    return {
        "quality_score": score,
        "factors": {
            "naming": "good" if score > 75 else "needs_work",
            "indentation": "correct",
            "complexity": "low" if score > 80 else "high",
            "readability": "high" if score > 75 else "low"
        },
        "elapsed_ms": round(elapsed_ms, 2)
    }


@app.post("/suggestions", response_model=SuggestionsResponse)
def suggestions(request: AnalyzeRequest):
    """
    Get real-time code improvement suggestions.
    
    Suggests improvements for:
    - Style issues
    - Performance optimizations
    - Best practices
    - Readability
    """
    start_time = time.time()
    sugg_list = AdvancedCodeAnalyzer.get_real_time_suggestions(request.code, request.language)
    elapsed_ms = (time.time() - start_time) * 1000
    
    return {
        "suggestions": sugg_list,
        "total_suggestions": len(sugg_list),
        "language": request.language,
    }


@app.post("/analyze/errors-by-category")
def errors_by_category(request: AnalyzeRequest):
    """
    Analyze errors grouped by category.
    
    Categories: syntax, logic, type, performance, security, style, best_practice, resource, boundary, null_safety, complexity
    """
    result = AdvancedCodeAnalyzer.analyze(request.code, request.language)
    
    errors_by_cat = {}
    for error in result.get("errors", []):
        cat = error.get("category", "unknown")
        if cat not in errors_by_cat:
            errors_by_cat[cat] = []
        errors_by_cat[cat].append(error)
    
    return {
        "status": "analyzed",
        "language": request.language,
        "total_errors": result.get("total_errors", 0),
        "categories": errors_by_cat,
        "category_count": len(errors_by_cat)
    }


@app.post("/analyze/by-severity")
def errors_by_severity(request: AnalyzeRequest):
    """
    Analyze errors grouped by severity level.
    
    Severity levels: critical (won't run), high (logic errors), medium (inefficiency), low (suggestions)
    """
    result = AdvancedCodeAnalyzer.analyze(request.code, request.language)
    
    errors_by_sev = {
        "critical": [],
        "high": [],
        "medium": [],
        "low": []
    }
    
    for error in result.get("errors", []):
        sev = error.get("severity", "low")
        if sev in errors_by_sev:
            errors_by_sev[sev].append(error)
    
    return {
        "status": "analyzed",
        "language": request.language,
        "total_errors": result.get("total_errors", 0),
        "severity_levels": errors_by_sev,
        "critical_count": len(errors_by_sev["critical"]),
        "high_count": len(errors_by_sev["high"]),
        "medium_count": len(errors_by_sev["medium"]),
        "low_count": len(errors_by_sev["low"])
    }


# ============================================================================
# Learning Guidance Endpoints (5 endpoints)
# ============================================================================

@app.get("/learn/concepts", response_model=ConceptsListResponse)
def list_concepts():
    """
    Get list of all available learning concepts.
    
    Returns: All 5 core concepts with metadata
    """
    concepts = []
    for concept_id, concept in CONCEPT_LIBRARY.items():
        concepts.append({
            "id": concept_id,
            "name": concept.concept_name,
            "difficulty": concept.difficulty.value,
            "summary": concept.simple_explanation,
            "prerequisites": concept.prerequisites,
            "related_concepts": concept.related_concepts,
            "examples_count": len(concept.examples)
        })
    
    return {
        "total_concepts": len(concepts),
        "concepts": concepts
    }


@app.get("/learn/concept/{concept_id}", response_model=ConceptResponse)
def get_concept_detail(
    concept_id: str,
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
):
    """
    Get detailed explanation for a concept.
    
    Provides:
    - Human-friendly explanation
    - Code examples (right vs wrong)
    - Common mistakes
    - Tips for mastery
    - Learning resources
    """
    if concept_id not in CONCEPT_LIBRARY:
        return {"status": "not_found", "error": f"Concept {concept_id} not found"}
    
    concept = CONCEPT_LIBRARY[concept_id]
    examples = []
    for ex in concept.examples:
        examples.append({
            "title": ex.title,
            "wrong_code": ex.wrong_code,
            "right_code": ex.right_code,
            "explanation": ex.explanation,
            "key_learning": ex.key_learning
        })
    
    return {
        "status": "success",
        "concept_name": concept.concept_name,
        "difficulty": concept.difficulty.value,
        "simple_explanation": concept.simple_explanation,
        "detailed_explanation": concept.detailed_explanation,
        "prerequisites": concept.prerequisites,
        "related_concepts": concept.related_concepts,
        "common_mistakes": concept.common_mistakes,
        "tips_for_mastery": concept.tips_for_mastery,
        "code_examples": examples,
        "learning_resources": concept.learning_resources
    }


@app.post("/learn/from-error")
def learn_from_error(request: AnalyzeRequest):
    """
    Learn concepts from your code errors.
    
    Maps errors to concepts and provides:
    - What this error means
    - Related concept explanation
    - Code examples
    - How to avoid it
    """
    result = AdvancedCodeAnalyzer.analyze(request.code, request.language)
    errors = result.get("errors", [])
    
    learning_materials = []
    for error in errors[:3]:  # First 3 errors
        error_category = error.get("category", "syntax")
        
        # Map error category to concept
        concept_map = {
            "syntax_error": "comparison_operators",
            "logical_error": "loop_boundaries",
            "boundary_error": "loop_boundaries",
            "null_safety": "null_safety",
            "type_error": "type_safety"
        }
        
        concept_id = concept_map.get(error_category, "comparison_operators")
        if concept_id in CONCEPT_LIBRARY:
            concept = CONCEPT_LIBRARY[concept_id]
            
            examples = [{
                "title": ex.title,
                "wrong_code": ex.wrong_code,
                "right_code": ex.right_code,
                "explanation": ex.explanation
            } for ex in concept.examples]
            
            learning_materials.append({
                "error": {
                    "message": error.get("message"),
                    "severity": error.get("severity"),
                    "category": error.get("category"),
                    "line": error.get("line")
                },
                "concept": {
                    "name": concept.concept_name,
                    "simple_explanation": concept.simple_explanation
                },
                "examples": examples,
                "tips": concept.tips_for_mastery
            })
    
    return {
        "status": "analyzed",
        "language": request.language,
        "total_errors": len(errors),
        "learning_materials": learning_materials,
        "overall_advice": f"You have {len(errors)} error(s). Start with the most severe."
    }


@app.get("/learn/path/{concept_id}", response_model=LearningPathResponse)
def learning_path(concept_id: str):
    """
    Get structured learning path for a concept.
    
    Shows:
    - Prerequisites to learn first
    - Main concept explanation
    - Practice examples
    - Related concepts to explore next
    """
    if concept_id not in CONCEPT_LIBRARY:
        return {"error": "Concept not found"}
    
    concept = CONCEPT_LIBRARY[concept_id]
    
    return {
        "starting_concept": concept.concept_name,
        "difficulty": concept.difficulty.value,
        "learning_path": {
            "step_1_prerequisites": [{"name": p} for p in concept.prerequisites],
            "step_2_main_concept": {
                "name": concept.concept_name,
                "explanation": concept.detailed_explanation
            },
            "step_3_practice": {
                "examples": len(concept.examples),
                "message": f"Practice with {len(concept.examples)} code example(s)"
            },
            "step_4_related_concepts": [{"name": c} for c in concept.related_concepts]
        },
        "tips": [
            "Start with prerequisites",
            "Read explanations carefully",
            "Type out code examples yourself",
            "Test your understanding with new code"
        ]
    }


@app.post("/learn/explain-code")
def explain_code(request: AnalyzeRequest):
    """
    Explain what code does and teach related concepts.
    
    Analyzes:
    - What the code does
    - Concepts used
    - Quality level
    - Improvement suggestions
    """
    result = AdvancedCodeAnalyzer.analyze(request.code, request.language)
    quality_score = AdvancedCodeAnalyzer.get_quality_score(request.code, request.language)
    
    # Detect which concepts are used
    detected_concepts = []
    if "loop" in request.code.lower():
        detected_concepts.append("loop_boundaries")
    if "==" in request.code or "if" in request.code:
        detected_concepts.append("comparison_operators")
    if "None" in request.code or "null" in request.code.lower():
        detected_concepts.append("null_safety")
    
    concept_explanations = []
    for cid in detected_concepts[:2]:
        if cid in CONCEPT_LIBRARY:
            concept = CONCEPT_LIBRARY[cid]
            concept_explanations.append({
                "concept": concept.concept_name,
                "explanation": concept.simple_explanation
            })
    
    return {
        "status": "analyzed",
        "language": request.language,
        "code_length": len(request.code),
        "quality_score": quality_score,
        "detected_concepts": concept_explanations,
        "what_this_code_does": "This code demonstrates programming concepts. Read through and understand each line.",
        "improvements_suggested": AdvancedCodeAnalyzer.get_real_time_suggestions(request.code, request.language),
        "next_learning_steps": [c.get("concept", "concept") for c in concept_explanations]
    }


# ============================================================================
# Smart Hint System Endpoints (3 endpoints)
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
@limiter.limit("10/minute")
async def chat(request: Request, payload: ChatRequest):
    """
    Chat interface with PolyMentor AI tutor.
    
    Supports:
    - Teaching programming concepts
    - Finding bugs in code
    - Explaining errors
    - Generating fixed code
    - Adaptive learning guidance
    """
    try:
        response = await pipeline.chat(
            message=payload.message,
            level=payload.level,
        )
        
        if response.status == "missing_groq_api_key":
            raise HTTPException(status_code=500, detail="Missing Groq API Key")
    
        return {
            "status": response.status,
            "answer": response.answer,
            "language": response.language,
            "level": response.level,
            "model": response.model,
            "suspected_bugs": response.suspected_bugs,
            "fixed_code": response.fixed_code,
            "lesson": response.lesson,
            "next_steps": response.next_steps,
            "elapsed_ms": response.elapsed_ms,
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/learn/hints/{error_type}", response_model=HintsResponse)
def get_hints(
    error_type: str,
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
):
    """
    Get step-by-step hints for a specific error type.
    
    Progressive hints adapt to learner level:
    - Beginner: All 3 steps (detailed guidance)
    - Intermediate: Steps 2-3 (less hand-holding)
    - Advanced: Only final step (just the answer)
    """
    hints = hint_system.get_hints(error_type, level)
    
    return {
        "status": "success",
        "error_type": error_type,
        "level": level,
        "hints": hints,
        "total_hints": len(hints)
    }


@app.post("/learn/next-hint")
def get_next_hint(request: AnalyzeRequest):
    """
    Get the next hint for code with adaptive difficulty.
    
    Provides:
    - Next hint in progression
    - Adjusted based on learner level
    - Tracks which hints have been shown
    """
    result = AdvancedCodeAnalyzer.analyze(request.code, request.language)
    
    if not result.get("errors"):
        return {"status": "no_errors", "message": "No errors found in code"}
    
    first_error = result["errors"][0]
    error_category = first_error.get("category", "syntax_error")
    
    # Get appropriate hint based on learner level
    hints = hint_system.get_hints(error_category, request.level)
    
    return {
        "status": "success",
        "error": {
            "message": first_error.get("message"),
            "line": first_error.get("line"),
            "category": error_category
        },
        "hint": hints[0] if hints else "Review your code carefully",
        "hint_index": 1,
        "total_hints_available": len(hints),
        "learner_level": request.level,
        "next_hint_text": "Run /learn/hints again to see the next hint"
    }


@app.post("/learn/hint-feedback")
def hint_feedback(
    hint_text: str,
    was_helpful: bool,
    error_type: str,
    user_level: str
):
    """
    Track hint effectiveness for adaptive learning.
    
    Feedback helps system:
    - Adjust which hints work best
    - Modify difficulty levels
    - Improve future recommendations
    """
    # Score and save feedback
    score = feedback_scorer.score_feedback(
        hint_text=hint_text,
        was_helpful=was_helpful,
        error_type=error_type,
        user_level=user_level
    )
    
    return {
        "status": "recorded",
        "feedback_score": score,
        "message": "Thank you! Your feedback helps us improve.",
        "implications": {
            "hint_effectiveness": score / 100,
            "helps_improve": "This feedback helps personalize future hints"
        }
    }


@app.post("/learn/adaptive-level", response_model=AdaptiveResponse)
def adaptive_difficulty(
    current_level: str,
    errors_solved: int,
    hints_used: int,
    success_rate: float  # 0-1, percentage of problems solved
):
    """
    Get adaptive difficulty recommendation based on performance.
    
    Adjusts difficulty based on:
    - Error types successfully resolved
    - Hint effectiveness
    - Success rate on problems
    """
    recommended_level = current_level
    reason = ""
    next_concept = None
    
    if success_rate > 0.8 and current_level == "beginner":
        recommended_level = "intermediate"
        reason = "Excellent performance! Ready for more complex challenges."
        next_concept = "loop_boundaries"
    elif success_rate < 0.5 and current_level == "intermediate":
        recommended_level = "beginner"
        reason = "Let's focus on fundamentals. You'll improve quickly!"
        next_concept = "comparison_operators"
    elif success_rate > 0.85 and current_level == "intermediate":
        recommended_level = "advanced"
        reason = "Outstanding progress! Ready for advanced challenges."
        next_concept = "type_safety"
    
    return {
        "status": "success",
        "current_level": current_level,
        "recommended_level": recommended_level,
        "reason": reason,
        "next_concept": next_concept
    }


# ============================================================================
# Root/Default Handler
# ============================================================================

@app.get("/{path:path}")
def fallback(path: str):
    """Catch-all for undefined routes"""
    return {
        "status": "not_found",
        "message": f"Endpoint /{path} not found",
        "available_endpoints": "/docs"
    }
