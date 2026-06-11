# PolyMentor: Complete Project Documentation v2.0
## With Groq API, MongoDB, MLOps, and Hugging Face Fine-Tuning

**Last Updated**: June 11, 2026  
**Version**: 2.0  
**Status**: Planning & Implementation Guide

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Current Implementation](#current-implementation)
3. [New Requirements & Architecture](#new-requirements--architecture)
4. [Technology Stack](#technology-stack)
5. [Data Flow Diagram](#data-flow-diagram)
6. [Detailed Implementation Guide](#detailed-implementation-guide)
7. [Groq API Integration](#groq-api-integration)
8. [MongoDB Setup & Configuration](#mongodb-setup--configuration)
9. [MLOps Pipeline](#mlops-pipeline)
10. [Hugging Face Fine-Tuning](#hugging-face-fine-tuning)
11. [Complete Setup Instructions](#complete-setup-instructions)
12. [API Endpoints Reference](#api-endpoints-reference)
13. [Troubleshooting & Best Practices](#troubleshooting--best-practices)

---

## 🎯 Project Overview

### What is PolyMentor?

PolyMentor is an **intelligent, AI-powered coding mentor** that:
- Teaches programming concepts at multiple difficulty levels (beginner, intermediate, advanced)
- Analyzes code across 4 languages (Python, JavaScript, C++, Java)
- Identifies and explains bugs with pedagogical approach
- Generates progressive hints (3-step system)
- Adapts difficulty based on learner performance
- Learns from user interactions via MLOps pipeline

### Core Mission

Provide a **patient, adaptive coding mentor** that explains concepts, not just fixes code.

---

## 📊 Current Implementation

### What Already Exists

#### 1. **Backend API (FastAPI)**
- **File**: `src/api/app.py`
- **Features**:
  - 17+ endpoints for error detection, learning guidance, hints
  - CORS enabled for web integration
  - Request/Response validation with Pydantic
  - Real-time code analysis

#### 2. **Advanced Code Analyzer**
- **File**: `src/analysis/advanced_analyzer.py`
- **Capabilities**:
  - Detects 50+ coding patterns
  - Categorizes errors into 11 types
  - 4 severity levels (critical, high, medium, low)
  - <5ms response time
  - Language support: Python, JavaScript, C++, Java

#### 3. **Smart Hint System**
- **File**: `src/reasoning_engine/hint_system.py`
- **Features**:
  - Progressive 3-step hints
  - Adaptive difficulty (beginner → intermediate → advanced)
  - 13+ error-type specific strategies
  - Template-based hint generation

#### 4. **Feedback & Analytics**
- **File**: `src/reasoning_engine/feedback_scorer.py`
- **Capabilities**:
  - Tracks hint effectiveness (0-100)
  - Learner session analytics
  - Difficulty recommendations
  - Performance aggregation

#### 5. **Concept Library**
- **File**: `src/learning/concept_guide.py`
- **Coverage**:
  - 5 core learning concepts
  - Teaching materials & explanations
  - Learning paths
  - Beginner/Intermediate/Advanced levels

#### 6. **Testing Suite**
- **File**: `tests/test_hint_system.py`
- **Coverage**: 28 comprehensive tests
- **Validated**: All tests passing

---

## 🚀 New Requirements & Architecture

### New Components to Implement

#### 1. **Groq API Integration**
**Purpose**: Fast LLM-based responses for complex explanations

**What it does**:
- User submits code with a question
- Groq generates detailed, pedagogical explanations
- Responses are stored + analyzed for fine-tuning data

**Example Flow**:
```
User: "Explain why my loop is infinite"
  ↓
[Groq API Call]
  ↓
Groq Response: "Your loop never changes X, so condition remains true..."
  ↓
[Store in MongoDB]
  ↓
[Use for fine-tuning dataset]
```

#### 2. **MongoDB Storage**
**Purpose**: Persistent data for training and analytics

**Data to Store**:
- User interactions (questions, code submitted)
- Groq responses (explanations, hints)
- Learner progress (errors solved, levels achieved)
- Model feedback (hint effectiveness, user ratings)
- Training dataset (code-explanation pairs)

#### 3. **MLOps Pipeline**
**Purpose**: Automatic model improvement based on data

**Workflow**:
- Collect user interactions → MongoDB
- Extract training data (successful explanations)
- Trigger fine-tuning job automatically
- Validate new model
- Deploy to production

#### 4. **Hugging Face Fine-Tuning**
**Purpose**: Custom model trained on your data

**Base Model**: CodeBERT or DistilBERT
**Task**: Generate better explanations for code errors
**Training Data**: User interactions + Groq responses
**Custom Parameters**:
- Learning rate: 2e-5
- Batch size: 16
- Epochs: 3-5
- Warmup steps: 500

---

## 🛠️ Technology Stack

### Current Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend | FastAPI 0.136.0 | REST API framework |
| Validation | Pydantic | Request/response validation |
| Analysis | Tree-Sitter | Code parsing |
| Testing | pytest | Test framework |
| Deployment | Docker | Containerization |

### New Stack Components

| Component | Technology | Purpose | Version |
|-----------|-----------|---------|---------|
| **LLM API** | Groq | Fast inference | Latest |
| **Database** | MongoDB | Data persistence | 7.0+ |
| **ML Framework** | Hugging Face Transformers | Model training | 4.36+ |
| **Training Framework** | PyTorch | Deep learning | 2.1+ |
| **MLOps Orchestration** | Apache Airflow / Prefect | Workflow automation | Latest |
| **Task Queue** | Celery + Redis | Async tasks | Latest |
| **Monitoring** | Prometheus + Grafana | Metrics & visualization | Latest |
| **Model Registry** | MLflow | Model versioning | 2.10+ |

### Database Schema (MongoDB)

```javascript
// Collections

// 1. users
{
  _id: ObjectId,
  user_id: String,
  name: String,
  level: String (beginner|intermediate|advanced),
  created_at: Date,
  updated_at: Date,
  metadata: {
    preferred_language: String,
    learning_style: String,
    total_problems_solved: Number
  }
}

// 2. user_interactions
{
  _id: ObjectId,
  user_id: String,
  timestamp: Date,
  code_submitted: String,
  language: String,
  question: String,
  groq_response: String,
  groq_model_used: String,
  latency_ms: Number,
  user_rating: Number (1-5),
  was_helpful: Boolean,
  tags: [String]
}

// 3. training_dataset
{
  _id: ObjectId,
  code_snippet: String,
  error_description: String,
  explanation_text: String,
  source: String (groq|predefined|user_feedback),
  language: String,
  difficulty_level: String,
  is_validated: Boolean,
  created_at: Date,
  used_in_training: Boolean
}

// 4. learner_progress
{
  _id: ObjectId,
  user_id: String,
  error_type: String,
  attempts: Number,
  successes: Number,
  avg_hints_used: Number,
  mastery_score: Number (0-100),
  last_practiced: Date,
  difficulty_level: String
}

// 5. model_versions
{
  _id: ObjectId,
  version: String,
  base_model: String,
  training_date: Date,
  training_samples: Number,
  validation_accuracy: Number,
  test_accuracy: Number,
  parameters: Object,
  is_active: Boolean,
  metrics: {
    bleu_score: Number,
    rouge_score: Number,
    custom_metric: Number
  }
}

// 6. mlops_jobs
{
  _id: ObjectId,
  job_id: String,
  job_type: String (training|evaluation|deployment),
  status: String (pending|running|completed|failed),
  start_time: Date,
  end_time: Date,
  config: Object,
  result: Object,
  error_message: String,
  logs_path: String
}
```

---

## 📈 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER INTERACTION                             │
│                  (Code submission + Question)                        │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   FastAPI App   │
                    │  (req handler)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼─────┐    ┌───────▼────────┐  ┌───────▼─────────┐
    │ Advanced  │    │   Groq API     │  │  Stored in      │
    │ Analyzer  │    │  (Fast LLM)    │  │  MongoDB        │
    │ (Local)   │    │  Explanation   │  │  (Interaction)  │
    └────┬─────┘    └───────┬────────┘  └───────┬─────────┘
         │                  │                    │
         │    ┌─────────────┘                    │
         │    │                                  │
    ┌────▼────▼──────┐                    ┌─────▼──────────┐
    │  Hint System   │                    │    MongoDB     │
    │  + Feedback    │                    │   (Storage)    │
    └────┬───────────┘                    └────────────────┘
         │                                       ▲
         │                            ┌──────────┘
         │                            │
         └────────┬─────────────────────────────┐
                  │                             │
          ┌───────▼──────┐           ┌──────────▼────────┐
          │   Response   │           │  Data Collection  │
          │   to User    │           │  Pipeline         │
          └──────────────┘           └──────────┬────────┘
                                                 │
                                        ┌────────▼────────┐
                                        │   MLOps Engine  │
                                        │   (Airflow)     │
                                        └────────┬────────┘
                                                 │
                    ┌────────────────────────────┼────────────────────┐
                    │                            │                    │
          ┌─────────▼──────────┐     ┌──────────▼─────┐    ┌─────────▼────┐
          │ Extract Training   │     │ Validate Data  │    │ Monitor Stats│
          │ Dataset            │     │ Quality        │    │              │
          └─────────┬──────────┘     └──────────┬─────┘    └──────────────┘
                    │                           │
                    └───────────────┬───────────┘
                                    │
                         ┌──────────▼──────────┐
                         │  Hugging Face      │
                         │  Fine-Tuning       │
                         │  (PyTorch)         │
                         └──────────┬─────────┘
                                    │
                         ┌──────────▼──────────┐
                         │ Model Validation    │
                         │ (Test on holdout)   │
                         └──────────┬─────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │                               │
         ┌──────────▼─────────┐     ┌──────────────▼──────┐
         │ Deploy to Prod     │     │ Archive Old Version │
         │ (Replace model)    │     │ in MLflow           │
         └────────────────────┘     └─────────────────────┘
```

---

## 🔧 Detailed Implementation Guide

### Phase 1: MongoDB Integration (Week 1)

#### Step 1: Install MongoDB

**Option A: Local Installation**
```bash
# Windows - using Chocolatey
choco install mongodb-community

# Or download from: https://www.mongodb.com/try/download/community

# Verify installation
mongod --version
```

**Option B: Docker (Recommended)**
```bash
# Pull MongoDB image
docker pull mongo:7.0

# Run MongoDB container
docker run -d \
  --name polymentor-mongo \
  -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=your_secure_password \
  -v mongo_data:/data/db \
  mongo:7.0
```

#### Step 2: Install Python MongoDB Driver

```bash
pip install pymongo motor
```

**Dependencies**:
- `pymongo`: Synchronous MongoDB driver
- `motor`: Async MongoDB driver for FastAPI

#### Step 3: Create Database Connection Module

**File**: `src/database/mongodb.py`

```python
import os
from typing import Optional
from motor.motor_asyncio import AsyncClient, AsyncDatabase
from pymongo import MongoClient

class MongoDBManager:
    """MongoDB connection and operations manager"""
    
    def __init__(self):
        self.uri = os.getenv(
            "MONGODB_URI",
            "mongodb+srv://user:password@cluster.mongodb.net/?retryWrites=true"
        )
        self.db_name = os.getenv("MONGODB_DB", "polymentor")
        self._client: Optional[AsyncClient] = None
        self._db: Optional[AsyncDatabase] = None
    
    async def connect(self):
        """Connect to MongoDB"""
        self._client = AsyncClient(self.uri)
        self._db = self._client[self.db_name]
        # Verify connection
        await self._client.admin.command('ping')
        print("✓ Connected to MongoDB")
    
    async def disconnect(self):
        """Disconnect from MongoDB"""
        if self._client:
            self._client.close()
            print("✓ Disconnected from MongoDB")
    
    @property
    def db(self) -> AsyncDatabase:
        """Get database instance"""
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db
    
    async def insert_interaction(self, interaction: dict):
        """Store user interaction"""
        result = await self.db.user_interactions.insert_one(interaction)
        return result.inserted_id
    
    async def get_user_history(self, user_id: str):
        """Fetch user interaction history"""
        return await self.db.user_interactions.find(
            {"user_id": user_id}
        ).to_list(100)
    
    async def insert_training_data(self, data: dict):
        """Store training data sample"""
        result = await self.db.training_dataset.insert_one(data)
        return result.inserted_id

# Global instance
mongo_manager = MongoDBManager()
```

#### Step 4: Update FastAPI App with MongoDB

**File**: `src/api/app.py` (additions)

```python
from fastapi import FastAPI
from contextlib import asynccontextmanager
from src.database.mongodb import mongo_manager
from datetime import datetime

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle - connect/disconnect DB"""
    # Startup
    await mongo_manager.connect()
    yield
    # Shutdown
    await mongo_manager.disconnect()

app = FastAPI(
    title="PolyMentor API",
    description="AI-powered coding mentor with Groq, MongoDB, MLOps",
    version="2.0.0",
    lifespan=lifespan
)

# ============ NEW ENDPOINT: Store Interaction ============
@app.post("/interactions/store")
async def store_interaction(
    user_id: str,
    code: str,
    language: str,
    question: str,
    groq_response: str,
    groq_model: str = "llama-3.1-70b-versatile",
    latency_ms: float = 0,
    rating: int = 0
):
    """Store user interaction for training dataset"""
    interaction = {
        "user_id": user_id,
        "timestamp": datetime.utcnow(),
        "code_submitted": code,
        "language": language,
        "question": question,
        "groq_response": groq_response,
        "groq_model_used": groq_model,
        "latency_ms": latency_ms,
        "user_rating": rating,
        "was_helpful": rating >= 4
    }
    
    result_id = await mongo_manager.insert_interaction(interaction)
    return {
        "status": "stored",
        "interaction_id": str(result_id),
        "timestamp": interaction["timestamp"]
    }

# ============ NEW ENDPOINT: Get User History ============
@app.get("/users/{user_id}/history")
async def get_user_interactions(user_id: str, limit: int = 50):
    """Get user's interaction history for analysis"""
    history = await mongo_manager.get_user_history(user_id)
    return {
        "user_id": user_id,
        "total_interactions": len(history),
        "recent_interactions": history[-limit:]
    }
```

---

### Phase 2: Groq API Integration (Week 1-2)

#### Step 1: Setup Groq API

**Get API Key**:
1. Visit: https://console.groq.com
2. Sign up/Login
3. Create API key
4. Store in environment variable: `GROQ_API_KEY`

**Install SDK**:
```bash
pip install groq
```

#### Step 2: Create Groq Integration Module

**File**: `src/llm/groq_integration.py`

```python
import os
from typing import Optional
from groq import Groq

class GroqAPIManager:
    """Manage Groq API interactions"""
    
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not set in environment")
        
        self.client = Groq(api_key=self.api_key)
        self.default_model = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")
        self.temperature = 0.7
        self.max_tokens = 1024
    
    def generate_explanation(
        self,
        code: str,
        error_message: str,
        question: str,
        language: str,
        level: str = "beginner",
        system_prompt: Optional[str] = None
    ) -> dict:
        """
        Generate pedagogical explanation using Groq
        
        Args:
            code: User's code snippet
            error_message: Error description
            question: User's question
            language: Programming language
            level: Difficulty level (beginner|intermediate|advanced)
            system_prompt: Custom system prompt
        
        Returns:
            {
                "explanation": str,
                "key_concepts": [str],
                "example": str,
                "next_steps": [str],
                "model": str,
                "latency_ms": float
            }
        """
        
        if system_prompt is None:
            system_prompt = self._get_system_prompt(level, language)
        
        prompt = f"""
Code Language: {language}
Difficulty Level: {level}
User's Code:
```{language}
{code}
```

Error Found: {error_message}

User's Question: {question}

Please provide:
1. A clear explanation of why this error occurred
2. The core concept being misunderstood
3. A corrected code example
4. Practice tips to prevent this error
        """.strip()
        
        import time
        start = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.default_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=0.95
            )
            
            latency = (time.time() - start) * 1000
            
            return {
                "explanation": response.choices[0].message.content,
                "model": self.default_model,
                "latency_ms": latency,
                "tokens_used": response.usage.total_tokens,
                "status": "success"
            }
        
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "latency_ms": (time.time() - start) * 1000
            }
    
    def _get_system_prompt(self, level: str, language: str) -> str:
        """Get appropriate system prompt for difficulty level"""
        
        if level == "beginner":
            return f"""You are a patient, encouraging coding mentor helping beginners learn {language}.
Your explanations should:
- Use simple language without jargon
- Explain the 'why' before the 'what'
- Provide relatable analogies
- Offer step-by-step fixes
- Encourage learning over just fixing
- Focus on core concepts
"""
        
        elif level == "intermediate":
            return f"""You are a knowledgeable {language} mentor for intermediate learners.
Your explanations should:
- Reference relevant concepts and patterns
- Explain efficiency and best practices
- Connect errors to larger design patterns
- Suggest improvements beyond the fix
"""
        
        else:  # advanced
            return f"""You are a senior {language} expert mentor.
Your explanations should:
- Discuss advanced patterns and optimizations
- Reference language specifications
- Explain edge cases and performance implications
- Suggest architectural improvements
"""

# Global instance
groq_manager = GroqAPIManager()
```

#### Step 3: Add Groq Endpoints to FastAPI

**File**: `src/api/app.py` (additions)

```python
from src.llm.groq_integration import groq_manager
from pydantic import BaseModel

class ExplanationRequest(BaseModel):
    code: str
    error_message: str
    question: str
    language: str
    level: str = "beginner"

@app.post("/explain/detailed")
async def get_detailed_explanation(request: ExplanationRequest):
    """Get Groq-powered detailed explanation with MongoDB storage"""
    
    # Generate explanation via Groq
    result = groq_manager.generate_explanation(
        code=request.code,
        error_message=request.error_message,
        question=request.question,
        language=request.language,
        level=request.level
    )
    
    if result["status"] == "success":
        # Store in MongoDB
        interaction = {
            "user_id": "anonymous",  # TODO: implement user auth
            "timestamp": datetime.utcnow(),
            "code_submitted": request.code,
            "language": request.language,
            "question": request.question,
            "groq_response": result["explanation"],
            "groq_model_used": result["model"],
            "latency_ms": result["latency_ms"],
            "tokens_used": result["tokens_used"]
        }
        
        await mongo_manager.insert_interaction(interaction)
    
    return result

@app.post("/explain/quick")
async def get_quick_explanation(
    code: str,
    language: str,
    level: str = "beginner"
):
    """Quick Groq explanation without storing"""
    result = groq_manager.generate_explanation(
        code=code,
        error_message="",
        question="Explain this code",
        language=language,
        level=level
    )
    return result
```

---

### Phase 3: MLOps Pipeline Setup (Week 2-3)

#### Step 1: Install MLOps Tools

```bash
# MLflow for model registry
pip install mlflow

# Airflow for workflow orchestration
pip install apache-airflow

# Or use Prefect (simpler for beginners)
pip install prefect

# Celery for task queue
pip install celery redis

# Monitoring
pip install prometheus-client
```

#### Step 2: Create MLOps Configuration

**File**: `src/mlops/config.py`

```python
import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class MLOpsConfig:
    """MLOps pipeline configuration"""
    
    # Data collection
    MIN_INTERACTIONS_FOR_TRAINING = 100
    DATA_VALIDATION_THRESHOLD = 0.8
    
    # Training
    TRAINING_BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    WARMUP_STEPS = 500
    MAX_SEQ_LENGTH = 512
    
    # Model
    BASE_MODEL = "distilbert-base-uncased"
    OUTPUT_DIR = "models_saved/huggingface_models"
    CHECKPOINT_DIR = "models_saved/checkpoints"
    
    # Validation
    MIN_VALIDATION_ACCURACY = 0.75
    TEST_SPLIT = 0.1
    VAL_SPLIT = 0.1
    
    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    MLFLOW_EXPERIMENT_NAME = "polymentor-finetuning"
    
    # Scheduling
    TRAINING_TRIGGER_HOUR = 2  # 2 AM UTC
    TRAINING_CHECK_INTERVAL = 86400  # Daily
    
    # Monitoring
    ENABLE_MONITORING = True
    PROMETHEUS_PORT = 8001

config = MLOpsConfig()
```

#### Step 3: Create Data Extraction Pipeline

**File**: `src/mlops/data_extraction.py`

```python
from typing import List, Dict, Any
from datetime import datetime, timedelta
from src.database.mongodb import mongo_manager

class TrainingDataExtractor:
    """Extract training data from MongoDB interactions"""
    
    async def extract_quality_interactions(
        self,
        days_back: int = 7,
        min_rating: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Extract high-quality interactions for training
        
        Criteria:
        - User rated helpful (rating >= 3)
        - Within last N days
        - Valid code snippet
        - Meaningful explanation
        """
        
        cutoff_date = datetime.utcnow() - timedelta(days=days_back)
        
        db = mongo_manager.db
        interactions = await db.user_interactions.find({
            "timestamp": {"$gte": cutoff_date},
            "user_rating": {"$gte": min_rating},
            "was_helpful": True,
            "code_submitted": {"$exists": True, "$ne": ""},
            "groq_response": {"$exists": True, "$ne": ""}
        }).to_list(None)
        
        return interactions
    
    async def format_for_training(
        self,
        interactions: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """
        Format interactions for fine-tuning
        
        Convert: {code, language, question, groq_response}
        To: {text_input, text_target}
        """
        
        training_samples = []
        
        for interaction in interactions:
            sample = {
                "text_input": f"""Code ({interaction.get('language', 'python')}):
{interaction.get('code_submitted', '')}

Question: {interaction.get('question', '')}

Provide a teaching explanation:""",
                
                "text_target": interaction.get('groq_response', ''),
                
                "metadata": {
                    "source": "groq",
                    "language": interaction.get('language'),
                    "user_rating": interaction.get('user_rating'),
                    "timestamp": str(interaction.get('timestamp'))
                }
            }
            training_samples.append(sample)
        
        return training_samples
    
    async def validate_dataset(
        self,
        samples: List[Dict[str, str]]
    ) -> tuple[List[Dict[str, str]], Dict[str, Any]]:
        """
        Validate training dataset quality
        
        Returns: (valid_samples, validation_stats)
        """
        
        stats = {
            "total_samples": len(samples),
            "valid_samples": 0,
            "invalid_samples": 0,
            "avg_input_length": 0,
            "avg_target_length": 0,
            "issues": []
        }
        
        valid = []
        
        for i, sample in enumerate(samples):
            input_text = sample.get("text_input", "")
            target_text = sample.get("text_target", "")
            
            # Validation checks
            if not input_text or len(input_text) < 20:
                stats["issues"].append(f"Sample {i}: Input too short")
                stats["invalid_samples"] += 1
                continue
            
            if not target_text or len(target_text) < 50:
                stats["issues"].append(f"Sample {i}: Target too short")
                stats["invalid_samples"] += 1
                continue
            
            if len(input_text) > 2000 or len(target_text) > 2000:
                stats["issues"].append(f"Sample {i}: Text too long")
                stats["invalid_samples"] += 1
                continue
            
            stats["valid_samples"] += 1
            valid.append(sample)
        
        # Calculate statistics
        if valid:
            stats["avg_input_length"] = sum(
                len(s["text_input"]) for s in valid
            ) / len(valid)
            stats["avg_target_length"] = sum(
                len(s["text_target"]) for s in valid
            ) / len(valid)
        
        return valid, stats
```

#### Step 4: Create Fine-Tuning Pipeline

**File**: `src/mlops/finetuning_pipeline.py`

```python
import os
import json
import torch
from typing import Dict, Any, Tuple
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer
)
from datasets import Dataset
import mlflow
from src.mlops.config import config
from src.database.mongodb import mongo_manager

class HuggingFaceFineTuner:
    """Fine-tune Hugging Face model on PolyMentor data"""
    
    def __init__(self, model_name: str = "t5-small"):
        """
        Initialize fine-tuner
        
        Args:
            model_name: Hugging Face model identifier
                Options:
                - "t5-small" (lightweight, fast)
                - "t5-base" (balanced)
                - "distilbert-base-uncased" (for classification)
                - "code-t5-base" (code-specific)
        """
        
        self.model_name = model_name
        self.tokenizer: PreTrainedTokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"✓ Using device: {self.device}")
        
        # MLflow setup
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    
    def load_model(self):
        """Load tokenizer and model"""
        print(f"Loading model: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        self.model.to(self.device)
        
        print(f"✓ Model loaded with {self.model.num_parameters():,} parameters")
    
    def preprocess_data(
        self,
        samples: list,
        max_input_length: int = 512,
        max_target_length: int = 256
    ) -> Dataset:
        """
        Convert samples to Hugging Face Dataset
        
        Args:
            samples: List of {"text_input": ..., "text_target": ...}
            max_input_length: Max tokens for input
            max_target_length: Max tokens for target
        
        Returns:
            Hugging Face Dataset object
        """
        
        def preprocess_function(examples):
            """Tokenize inputs and targets"""
            inputs = examples["text_input"]
            targets = examples["text_target"]
            
            # Tokenize inputs
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_input_length,
                truncation=True,
                padding="max_length"
            )
            
            # Tokenize targets
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    targets,
                    max_length=max_target_length,
                    truncation=True,
                    padding="max_length"
                )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Create dataset
        dataset = Dataset.from_dict({
            "text_input": [s["text_input"] for s in samples],
            "text_target": [s["text_target"] for s in samples]
        })
        
        # Apply tokenization
        tokenized = dataset.map(preprocess_function, batched=True)
        
        return tokenized
    
    def train(
        self,
        training_data: Dataset,
        eval_data: Dataset,
        num_epochs: int = config.NUM_EPOCHS,
        learning_rate: float = config.LEARNING_RATE
    ) -> Dict[str, Any]:
        """
        Fine-tune model
        
        Args:
            training_data: Tokenized training dataset
            eval_data: Tokenized evaluation dataset
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        
        Returns:
            Training results
        """
        
        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=config.OUTPUT_DIR,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=config.TRAINING_BATCH_SIZE,
            per_device_eval_batch_size=config.TRAINING_BATCH_SIZE,
            warmup_steps=config.WARMUP_STEPS,
            weight_decay=0.01,
            learning_rate=learning_rate,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            push_to_hub=False,
            logging_steps=50
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            model=self.model
        )
        
        # Trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=training_data,
            eval_dataset=eval_data,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train with MLflow logging
        with mlflow.start_run():
            mlflow.log_params({
                "model": self.model_name,
                "epochs": num_epochs,
                "learning_rate": learning_rate,
                "batch_size": config.TRAINING_BATCH_SIZE,
                "warmup_steps": config.WARMUP_STEPS
            })
            
            results = trainer.train()
            
            # Log metrics
            mlflow.log_metrics({
                "final_train_loss": results.training_loss,
            })
            
            # Save model
            self.save_model(f"polymentor-{datetime.now().isoformat()}")
            
            return results
    
    def save_model(self, version: str):
        """Save fine-tuned model and tokenizer"""
        save_path = f"{config.OUTPUT_DIR}/{version}"
        os.makedirs(save_path, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print(f"✓ Model saved to {save_path}")
        
        # Also save config
        config_path = f"{save_path}/training_config.json"
        with open(config_path, "w") as f:
            json.dump({
                "model_name": self.model_name,
                "saved_at": datetime.now().isoformat(),
                "base_model": self.model_name,
                "custom_params": {
                    "learning_rate": config.LEARNING_RATE,
                    "batch_size": config.TRAINING_BATCH_SIZE,
                    "epochs": config.NUM_EPOCHS
                }
            }, f, indent=2)
```

---

### Phase 4: Hugging Face Integration (Week 3)

#### Step 1: Create Model Inference Module

**File**: `src/models/huggingface_inference.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class HuggingFaceModel:
    """Load and use fine-tuned Hugging Face model"""
    
    def __init__(self, model_path: str):
        """
        Load model from path
        
        Args:
            model_path: Path to saved model directory
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        self.model.to(self.device)
    
    def generate_explanation(
        self,
        code: str,
        question: str,
        max_length: int = 256
    ) -> str:
        """
        Generate explanation for code
        
        Args:
            code: Code snippet
            question: User question
            max_length: Max output length
        
        Returns:
            Generated explanation
        """
        
        input_text = f"Code: {code}\nQuestion: {question}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        ).to(self.device)
        
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )
        
        explanation = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        
        return explanation
```

#### Step 2: Create Model Selection Logic

**File**: `src/models/model_selector.py`

```python
import os
from enum import Enum
from src.models.huggingface_inference import HuggingFaceModel

class ModelType(Enum):
    GROQ = "groq"  # Fast LLM
    HUGGINGFACE = "huggingface"  # Fine-tuned local model
    ENSEMBLE = "ensemble"  # Both

class ModelSelector:
    """Select best model for explanation"""
    
    def __init__(self):
        self.groq_available = os.getenv("GROQ_API_KEY") is not None
        self.hf_model = None
        self.load_huggingface_model()
    
    def load_huggingface_model(self):
        """Load latest fine-tuned model"""
        model_dir = "models_saved/huggingface_models"
        if os.path.exists(model_dir):
            # Load latest version
            versions = os.listdir(model_dir)
            if versions:
                latest = sorted(versions)[-1]
                try:
                    self.hf_model = HuggingFaceModel(f"{model_dir}/{latest}")
                    print(f"✓ Loaded HF model: {latest}")
                except Exception as e:
                    print(f"✗ Failed to load HF model: {e}")
    
    def select_model(
        self,
        response_quality: str = "balanced"  # fast|balanced|best
    ) -> ModelType:
        """
        Select best model based on requirements
        
        Args:
            response_quality: Quality vs speed tradeoff
        
        Returns:
            Model type to use
        """
        
        if response_quality == "fast":
            return ModelType.GROQ if self.groq_available else ModelType.HUGGINGFACE
        
        elif response_quality == "best":
            return ModelType.GROQ  # Groq is more powerful
        
        else:  # balanced
            if self.groq_available and self.hf_model:
                return ModelType.ENSEMBLE
            return ModelType.GROQ if self.groq_available else ModelType.HUGGINGFACE
```

---

### Phase 5: Deployment & Monitoring (Week 4)

#### Step 1: Create Docker Compose for Full Stack

**File**: `docker-compose-full.yml`

```yaml
version: '3.8'

services:
  # MongoDB
  mongodb:
    image: mongo:7.0
    container_name: polymentor-mongo
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: secure_password_here
      MONGO_INITDB_DATABASE: polymentor
    volumes:
      - mongo_data:/data/db
    networks:
      - polymentor

  # Redis (for Celery)
  redis:
    image: redis:7-alpine
    container_name: polymentor-redis
    ports:
      - "6379:6379"
    networks:
      - polymentor

  # FastAPI Application
  app:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: polymentor-api
    ports:
      - "8000:8000"
    environment:
      MONGODB_URI: mongodb://admin:secure_password_here@mongodb:27017/polymentor
      GROQ_API_KEY: ${GROQ_API_KEY}
      MLFLOW_TRACKING_URI: http://mlflow:5000
      REDIS_URL: redis://redis:6379/0
    depends_on:
      - mongodb
      - redis
    networks:
      - polymentor
    command: uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload

  # MLflow Server
  mlflow:
    image: ghcr.io/mlflow/mlflow:v2.10.0
    container_name: polymentor-mlflow
    ports:
      - "5000:5000"
    environment:
      BACKEND_STORE_URI: postgresql://mlflow:mlflow@postgres:5432/mlflow
      ARTIFACT_ROOT: /mlflow/artifacts
    volumes:
      - mlflow_artifacts:/mlflow/artifacts
    depends_on:
      - postgres
    networks:
      - polymentor
    command: mlflow server --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow --default-artifact-root /mlflow/artifacts --host 0.0.0.0

  # PostgreSQL (for MLflow)
  postgres:
    image: postgres:15-alpine
    container_name: polymentor-postgres
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - polymentor

  # Prometheus (Monitoring)
  prometheus:
    image: prom/prometheus:latest
    container_name: polymentor-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    networks:
      - polymentor
    command: --config.file=/etc/prometheus/prometheus.yml

  # Grafana (Visualization)
  grafana:
    image: grafana/grafana:latest
    container_name: polymentor-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: admin
    volumes:
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - polymentor

volumes:
  mongo_data:
  mlflow_artifacts:
  postgres_data:
  prometheus_data:
  grafana_data:

networks:
  polymentor:
    driver: bridge
```

#### Step 2: Create Monitoring Configuration

**File**: `monitoring/prometheus.yml`

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'polymentor-api'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: '/metrics'

  - job_name: 'mlflow'
    static_configs:
      - targets: ['localhost:5000']

  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
```

---

## 🔌 Groq API Integration

### API Reference

#### Available Models

| Model | Speed | Quality | Cost | Best For |
|-------|-------|---------|------|----------|
| `llama-3.1-70b-versatile` | Fast | High | Low | Default, good balance |
| `llama-3.3-70b-versatile` | Very Fast | High | Very Low | Production |
| `mixtral-8x7b-32768` | Medium | Very High | Medium | Complex explanations |

### Request Rate Limits

- **Tier 1 (Free)**: 30 requests/minute
- **Tier 2 (Paid)**: 600 requests/minute
- **Tier 3 (Enterprise)**: Custom

### Error Handling

```python
try:
    response = groq_manager.generate_explanation(...)
except ValueError as e:
    # Invalid input
    return {"error": str(e), "status": "invalid_input"}
except ConnectionError as e:
    # Network error
    return {"error": "Groq API unavailable", "status": "unavailable"}
except Exception as e:
    # Unexpected error
    return {"error": str(e), "status": "error"}
```

---

## 💾 MongoDB Setup & Configuration

### Collections Overview

```javascript
// Create indexes for performance
db.user_interactions.createIndex({ user_id: 1, timestamp: -1 })
db.user_interactions.createIndex({ timestamp: 1 }, { expireAfterSeconds: 7776000 }) // 90 days TTL
db.training_dataset.createIndex({ is_validated: 1 })
db.learner_progress.createIndex({ user_id: 1 })
db.model_versions.createIndex({ is_active: 1 })
db.mlops_jobs.createIndex({ status: 1, start_time: -1 })
```

### Backup Strategy

```bash
# Daily backup
mongodump --uri "mongodb+srv://user:pass@cluster.mongodb.net" --out /backups/polymentor_$(date +%Y%m%d)

# Restore from backup
mongorestore --uri "mongodb+srv://user:pass@cluster.mongodb.net" /backups/polymentor_20260611
```

---

## 🤖 MLOps Pipeline

### Training Job Scheduler

**File**: `src/mlops/scheduler.py`

```python
from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime, time
from src.mlops.data_extraction import TrainingDataExtractor
from src.mlops.finetuning_pipeline import HuggingFaceFineTuner

class TrainingScheduler:
    """Schedule automatic training jobs"""
    
    def __init__(self):
        self.scheduler = BackgroundScheduler()
    
    def start(self):
        """Start scheduler"""
        # Schedule daily training at 2 AM UTC
        self.scheduler.add_job(
            self.run_training_pipeline,
            'cron',
            hour=2,
            minute=0,
            timezone='UTC'
        )
        self.scheduler.start()
        print("✓ Training scheduler started")
    
    async def run_training_pipeline(self):
        """Execute full training pipeline"""
        try:
            print("\n" + "="*60)
            print(f"Starting training pipeline at {datetime.utcnow()}")
            print("="*60)
            
            # Step 1: Extract data
            extractor = TrainingDataExtractor()
            interactions = await extractor.extract_quality_interactions(days_back=7)
            
            if len(interactions) < 50:
                print(f"✗ Not enough data ({len(interactions)} < 50)")
                return
            
            print(f"✓ Extracted {len(interactions)} interactions")
            
            # Step 2: Format data
            samples = await extractor.format_for_training(interactions)
            print(f"✓ Formatted {len(samples)} training samples")
            
            # Step 3: Validate
            valid_samples, stats = await extractor.validate_dataset(samples)
            print(f"✓ Validated: {stats['valid_samples']}/{stats['total_samples']} samples")
            
            if stats['valid_samples'] < 50:
                print("✗ Not enough valid samples")
                return
            
            # Step 4: Train
            finetuner = HuggingFaceFineTuner(model_name="t5-small")
            finetuner.load_model()
            
            # Split data
            import random
            random.shuffle(valid_samples)
            split = int(0.9 * len(valid_samples))
            train_data = valid_samples[:split]
            eval_data = valid_samples[split:]
            
            # Preprocess
            train_dataset = finetuner.preprocess_data(train_data)
            eval_dataset = finetuner.preprocess_data(eval_data)
            
            # Train
            results = finetuner.train(train_dataset, eval_dataset)
            
            print(f"✓ Training complete!")
            print(f"  Final loss: {results.training_loss:.4f}")
            
        except Exception as e:
            print(f"✗ Training failed: {e}")
```

---

## 📚 Hugging Face Fine-Tuning

### Model Selection Guide

| Task | Base Model | Why |
|------|-----------|-----|
| **Code explanation** | `t5-small` or `t5-base` | Designed for text-to-text, good for QA |
| **Error classification** | `distilbert-base-uncased` | Fast classification, good accuracy |
| **Code-specific** | `code-t5-base` | Trained on code, understands structure |
| **High quality** | `t5-large` | 770M parameters, best results |

### Training Parameters

```python
# Default parameters (optimized for PolyMentor)
LEARNING_RATE = 2e-5          # Standard for fine-tuning
BATCH_SIZE = 16               # Adjust down if CUDA OOM
EPOCHS = 3                    # Usually 2-5 for fine-tuning
WARMUP_STEPS = 500            # 10% of training steps
MAX_SEQ_LENGTH = 512          # T5 standard
DROPOUT = 0.1                 # Prevent overfitting

# For limited data (<1000 samples):
LEARNING_RATE = 5e-5          # Lower LR
EPOCHS = 5                    # More epochs
WEIGHT_DECAY = 0.01           # L2 regularization

# For large data (>10000 samples):
LEARNING_RATE = 2e-5          # Standard
EPOCHS = 2                    # Less overfitting
BATCH_SIZE = 32               # Larger batches
```

### Evaluation Metrics

```python
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu

def evaluate_model(predictions, references):
    """Evaluate generated explanations"""
    
    # BLEU score (matches reference translations)
    bleu = corpus_bleu(
        [[ref.split()] for ref in references],
        [pred.split() for pred in predictions]
    )
    
    # ROUGE score (overlap with reference)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    rouge_scores = []
    for pred, ref in zip(predictions, references):
        score = scorer.score(ref, pred)
        rouge_scores.append(score['rougeL'].fmeasure)
    
    return {
        "bleu": bleu,
        "rouge_l": sum(rouge_scores) / len(rouge_scores),
        "avg_prediction_length": sum(len(p.split()) for p in predictions) / len(predictions)
    }
```

---

## 🚀 Complete Setup Instructions

### Quick Start (All Components)

```bash
# 1. Clone and setup
git clone https://github.com/OkashaRehman/PolyMentor.git
cd PolyMentor

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install groq motor pymongo mlflow transformers torch

# 4. Set environment variables
export GROQ_API_KEY="your_groq_api_key"
export MONGODB_URI="mongodb+srv://user:pass@cluster.mongodb.net"
export MLFLOW_TRACKING_URI="http://localhost:5000"

# 5. Start MongoDB
docker run -d --name polymentor-mongo -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=password \
  mongo:7.0

# 6. Start MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db

# 7. Start FastAPI
uvicorn src.api.app:app --reload

# 8. Initialize training scheduler
python scripts/start_scheduler.py
```

### Docker Compose (Recommended)

```bash
# Start entire stack
docker-compose -f docker-compose-full.yml up -d

# Check status
docker-compose -f docker-compose-full.yml ps

# View logs
docker-compose -f docker-compose-full.yml logs -f app

# Stop stack
docker-compose -f docker-compose-full.yml down
```

---

## 📡 API Endpoints Reference

### Health & Info

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check API status |
| `/info` | GET | Get API version & features |

### Code Analysis (Existing)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/analyze/basic` | POST | Quick code analysis |
| `/analyze/detailed` | POST | Detailed error analysis |
| `/languages` | GET | Supported languages |

### Groq Integration (New)

| Endpoint | Method | Purpose | Request |
|----------|--------|---------|---------|
| `/explain/detailed` | POST | Get detailed Groq explanation | `code, error_message, question, language, level` |
| `/explain/quick` | POST | Quick Groq explanation | `code, language, level` |

### MongoDB Storage (New)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/interactions/store` | POST | Store interaction |
| `/users/{user_id}/history` | GET | Get user history |
| `/training-data/quality` | GET | Get training dataset quality stats |

### MLOps (New)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/mlops/status` | GET | Training pipeline status |
| `/mlops/trigger-training` | POST | Manually trigger training |
| `/mlops/models` | GET | List all model versions |

### Monitoring (New)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/metrics` | GET | Prometheus metrics |
| `/health/detailed` | GET | Detailed health info |

---

## 🛡️ Troubleshooting & Best Practices

### Common Issues & Solutions

#### 1. **MongoDB Connection Fails**
```python
# ✗ Error: Server selection timed out
# Solution: Check MongoDB is running
docker ps | grep mongo

# Check connection string
echo $MONGODB_URI
```

#### 2. **Groq API Rate Limited**
```python
# ✗ Error: 429 Too Many Requests
# Solution: Implement backoff
import time
from tenacity import retry, wait_exponential

@retry(wait=wait_exponential(multiplier=1, min=2, max=10))
def call_groq_with_retry(...):
    return groq_manager.generate_explanation(...)
```

#### 3. **CUDA Out of Memory**
```bash
# Solution: Reduce batch size
TRAINING_BATCH_SIZE=8  # Down from 16
INFERENCE_BATCH_SIZE=4

# Or use CPU
export CUDA_VISIBLE_DEVICES=""
```

#### 4. **Model Takes Too Long to Load**
```python
# Solution: Use quantized model
from transformers import AutoModelForSeq2SeqLM
model = AutoModelForSeq2SeqLM.from_pretrained(
    "t5-small",
    load_in_8bit=True,  # 8-bit quantization
    device_map="auto"
)
```

### Best Practices

#### 1. **Data Quality**
- Validate all training data before using
- Remove duplicates and very similar samples
- Keep ratio of languages balanced
- Remove personally identifiable information

#### 2. **Training Safety**
- Always maintain validation set (10%)
- Monitor for overfitting (train loss << eval loss)
- Use early stopping
- Save checkpoints frequently

#### 3. **Monitoring**
- Set up alerts for failed training jobs
- Monitor MongoDB disk usage
- Track API latency and errors
- Review model quality metrics weekly

#### 4. **Security**
- Never commit API keys
- Use environment variables
- Encrypt sensitive data in MongoDB
- Implement rate limiting
- Use authentication for all endpoints

#### 5. **Performance**
- Cache model in memory
- Use async/await for I/O
- Index frequently queried MongoDB fields
- Batch API calls when possible

---

## 📊 Monitoring & Analytics Dashboard

### Key Metrics to Track

```python
# Created file: src/monitoring/metrics.py

from prometheus_client import Counter, Histogram, Gauge

# Request metrics
api_requests = Counter(
    'polymentor_api_requests_total',
    'Total API requests',
    ['method', 'endpoint']
)

api_latency = Histogram(
    'polymentor_api_latency_seconds',
    'API latency in seconds',
    ['endpoint']
)

# Groq metrics
groq_calls = Counter(
    'polymentor_groq_calls_total',
    'Total Groq API calls',
    ['model']
)

groq_latency = Histogram(
    'polymentor_groq_latency_ms',
    'Groq latency in milliseconds'
)

# Training metrics
training_runs = Counter(
    'polymentor_training_runs_total',
    'Total training runs',
    ['status']
)

model_accuracy = Gauge(
    'polymentor_model_accuracy',
    'Current model accuracy',
    ['model_version']
)

# MongoDB metrics
db_operations = Counter(
    'polymentor_db_operations_total',
    'Total DB operations',
    ['operation', 'collection']
)

dataset_size = Gauge(
    'polymentor_training_dataset_size',
    'Size of training dataset'
)
```

### Grafana Dashboard Setup

1. Open Grafana: `http://localhost:3000`
2. Add Prometheus data source: `http://prometheus:9090`
3. Create dashboard with panels:
   - API Requests/sec
   - Average Latency
   - Groq API Cost
   - Training Success Rate
   - Model Accuracy Trend
   - MongoDB Storage Usage

---

## 📝 Summary & Next Steps

### What's Implemented
✅ Advanced code analyzer (Python, JS, C++, Java)
✅ Smart hint system (3-step progressive)
✅ Feedback scoring & analytics
✅ FastAPI backend with 17+ endpoints
✅ Comprehensive test suite (28 tests)

### What Needs Implementation
1. **MongoDB Integration** (Phase 1)
   - Database setup
   - Connection management
   - Data models
   
2. **Groq API Integration** (Phase 2)
   - API wrapper
   - Explanation generation
   - Error handling
   
3. **MLOps Pipeline** (Phase 3)
   - Data extraction
   - Training automation
   - Model registry
   
4. **Hugging Face Fine-Tuning** (Phase 4)
   - Model loading
   - Data preprocessing
   - Training & evaluation
   
5. **Deployment & Monitoring** (Phase 5)
   - Docker setup
   - Health checks
   - Prometheus metrics

### Timeline
- **Phase 1**: 3-4 days
- **Phase 2**: 3-4 days
- **Phase 3**: 4-5 days
- **Phase 4**: 3-4 days
- **Phase 5**: 2-3 days

**Total**: 15-20 days for complete implementation

---

## 📞 Support & Resources

### Documentation References
- [Groq API Docs](https://console.groq.com/docs)
- [MongoDB Documentation](https://docs.mongodb.com)
- [Hugging Face Guide](https://huggingface.co/docs/transformers)
- [MLflow Documentation](https://mlflow.org/docs)
- [FastAPI Guide](https://fastapi.tiangolo.com)

### Useful Commands

```bash
# MongoDB
mongosh --uri "mongodb://admin:password@localhost:27017"
db.user_interactions.countDocuments()
db.training_dataset.find().limit(5)

# MLflow
mlflow ui
mlflow models predict -m models:/polymentor/production -i input.json

# Training status
curl http://localhost:8000/mlops/status

# Monitor logs
docker logs polymentor-api -f
```

---

**Document Version**: 2.0  
**Last Updated**: June 11, 2026  
**Status**: Complete Implementation Guide  
**Next Review**: After Phase 1 completion
