# PolyMentor: Quick Reference Guide

**For Developers Building the System**

---

## 🚀 Quick Start Commands

### Initial Setup (First Time)

```bash
# 1. Clone repository
git clone https://github.com/OkashaRehman/PolyMentor.git
cd PolyMentor

# 2. Create Python environment (Python 3.10+)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install core dependencies
pip install -r requirements.txt

# 4. Install new components (for v2.0)
pip install groq motor pymongo mlflow transformers torch pandas

# 5. Setup environment file
cp .env.example .env
# Edit .env with your API keys and passwords

# 6. Start MongoDB (Docker recommended)
docker run -d --name polymentor-mongo -p 27017:27017 \
  -e MONGO_INITDB_ROOT_USERNAME=admin \
  -e MONGO_INITDB_ROOT_PASSWORD=your_password \
  mongo:7.0

# 7. Start FastAPI server
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000

# 8. Open API docs
# Navigate to: http://localhost:8000/docs
```

### Daily Development

```bash
# Activate environment
source venv/bin/activate

# Run tests
pytest tests/ -v

# Start dev server
python scripts/run_dev.py

# Check MongoDB
mongosh --uri "mongodb://admin:password@localhost:27017/polymentor"

# View MLflow
mlflow ui  # Opens http://localhost:5000

# Check API health
curl http://localhost:8000/health
```

### Full Stack (All Components)

```bash
# One-liner to start everything
docker-compose -f docker-compose-full.yml up -d

# Check status
docker-compose -f docker-compose-full.yml ps

# View logs
docker-compose -f docker-compose-full.yml logs -f app

# Stop everything
docker-compose -f docker-compose-full.yml down
```

---

## 📚 Architecture Overview

### System Components

```
┌─────────────────────────────────────────┐
│         USER INTERFACE                  │
│     (Web/CLI/Mobile App)                │
└──────────────┬──────────────────────────┘
               │ HTTP/REST
┌──────────────▼──────────────────────────┐
│      FastAPI Backend (Port 8000)        │
│  ┌────────────────────────────────────┐ │
│  │  Advanced Code Analyzer            │ │
│  │  - Tree-sitter parsing             │ │
│  │  - 50+ patterns detected           │ │
│  │  - 11 error categories             │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Smart Hint System                 │ │
│  │  - 3-step progressive hints        │ │
│  │  - Adaptive difficulty             │ │
│  └────────────────────────────────────┘ │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │  Groq API Integration              │ │
│  │  - LLM explanations                │ │
│  │  - Fast inference                  │ │
│  └────────────────────────────────────┘ │
└──────────────┬──────────────┬────────────┘
               │              │
        ┌──────▼──────┐  ┌────▼──────────┐
        │  MongoDB    │  │  Groq API    │
        │  (Data)     │  │  (Cloud)     │
        └──────┬──────┘  └──────────────┘
               │
        ┌──────▼──────────────────────┐
        │   MLOps Pipeline (Phase 3)  │
        │  ┌──────────────────────┐   │
        │  │ Data Extraction      │   │
        │  │ Training Scheduler   │   │
        │  │ Job Management       │   │
        │  └──────────────────────┘   │
        └──────┬───────────────────────┘
               │
        ┌──────▼──────────────────────┐
        │ Hugging Face (Phase 4)      │
        │ ┌──────────────────────┐    │
        │ │ Fine-tuning Pipeline │    │
        │ │ Model Inference      │    │
        │ └──────────────────────┘    │
        └─────────────────────────────┘
```

---

## 🔧 Key Files & Their Purposes

### Backend API

| File | Purpose | Status |
|------|---------|--------|
| `src/api/app.py` | Main FastAPI application | ✅ Complete |
| `src/analysis/advanced_analyzer.py` | Code pattern detection | ✅ Complete |
| `src/reasoning_engine/hint_system.py` | Progressive hint generation | ✅ Complete |
| `src/reasoning_engine/feedback_scorer.py` | Analytics & tracking | ✅ Complete |
| `src/learning/concept_guide.py` | Teaching materials | ✅ Complete |

### New Components (v2.0)

| File | Purpose | Status | Phase |
|------|---------|--------|-------|
| `src/database/mongodb.py` | DB connection | 🔄 Planning | 1 |
| `src/llm/groq_integration.py` | Groq API wrapper | 🔄 Planning | 2 |
| `src/mlops/data_extraction.py` | Training data pipeline | 🔄 Planning | 3 |
| `src/mlops/finetuning_pipeline.py` | Model fine-tuning | 🔄 Planning | 4 |
| `src/models/huggingface_inference.py` | Model inference | 🔄 Planning | 4 |
| `src/monitoring/metrics.py` | Prometheus metrics | 🔄 Planning | 5 |

**Status**: ✅ Complete | 🔄 In Progress | 📋 Planned

---

## 📡 API Endpoints Quick Reference

### Existing Endpoints (v1.0)

```
GET    /health                      → Health check
GET    /languages                   → List supported languages
POST   /analyze/basic               → Quick code analysis
POST   /analyze/detailed            → Detailed analysis
GET    /learn/concepts              → List learning concepts
POST   /learn/explain               → Explain concept
GET    /learn/hints/{error_type}    → Get hint templates
POST   /learn/next-hint             → Next hint in progression
POST   /learn/hint-feedback         → Rate hint
POST   /learn/adaptive-level        → Get difficulty recommendation
```

### New Endpoints (v2.0)

```
PHASE 2 (Groq Integration):
POST   /explain/detailed            → Groq explanation
POST   /explain/quick               → Quick Groq explanation
GET    /groq/status                 → Check Groq API health

PHASE 1 (MongoDB):
POST   /interactions/store          → Store user interaction
GET    /users/{user_id}/history     → Get interaction history
GET    /training-data/quality       → Dataset quality stats

PHASE 3 (MLOps):
GET    /mlops/status                → Training pipeline status
POST   /mlops/trigger-training      → Manual trigger
GET    /mlops/jobs/{job_id}         → Job details
GET    /mlops/models                → List models
DELETE /mlops/jobs/{job_id}         → Cancel job

PHASE 5 (Monitoring):
GET    /health/detailed             → Detailed health info
GET    /metrics                     → Prometheus metrics
```

---

## 💾 MongoDB Quick Reference

### Connection String
```
# Local
mongodb://admin:password@localhost:27017/polymentor

# MongoDB Atlas (Cloud)
mongodb+srv://user:password@cluster.mongodb.net/polymentor?retryWrites=true
```

### Collections & Documents

```javascript
// Connect
mongosh "mongodb://admin:password@localhost:27017/polymentor"

// Switch database
use polymentor

// View collections
show collections

// Check document count
db.user_interactions.countDocuments()
db.training_dataset.countDocuments()

// Find documents
db.user_interactions.findOne({ user_id: "user123" })
db.training_dataset.find({ is_validated: true }).limit(5)

// Update document
db.user_interactions.updateOne(
  { _id: ObjectId("...") },
  { $set: { user_rating: 5 } }
)

// Delete old data (>90 days)
db.user_interactions.deleteMany({
  timestamp: { $lt: new Date(Date.now() - 90*24*60*60*1000) }
})

// Create indexes
db.user_interactions.createIndex({ user_id: 1, timestamp: -1 })
db.training_dataset.createIndex({ is_validated: 1 })

// Backup
mongodump --uri "mongodb://..." --out /backups/polymentor

// Restore
mongorestore --uri "mongodb://..." /backups/polymentor
```

---

## 🔑 Environment Variables

### Required Variables

```bash
# Groq
GROQ_API_KEY=gsk_xxxxxxxxxxxx
GROQ_MODEL=llama-3.1-70b-versatile

# MongoDB
MONGODB_URI=mongodb+srv://user:password@cluster.mongodb.net
MONGODB_DB=polymentor

# MLflow
MLFLOW_TRACKING_URI=http://localhost:5000

# API
API_HOST=0.0.0.0
API_PORT=8000
API_DEBUG=false

# Security
SECRET_KEY=your_secret_key_here
ALLOWED_ORIGINS=["*"]

# Training
TRAINING_BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=3

# Monitoring
PROMETHEUS_PORT=8001
ENABLE_ALERTS=true
```

### Setup .env File

```bash
# Copy template
cp .env.example .env

# Edit .env with your values
nano .env

# Verify loading
python -c "import os; print(os.getenv('GROQ_API_KEY'))"
```

---

## 🧪 Testing Commands

### Run All Tests
```bash
# Run everything
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test file
pytest tests/test_hint_system.py -v

# Specific test
pytest tests/test_hint_system.py::TestProgressiveHints::test_beginner_hints -v
```

### Test Database Operations
```bash
# Unit tests
pytest tests/test_mongodb_integration.py -v

# Load testing
locust -f tests/load_test.py --host=http://localhost:8000
```

### Manual API Testing
```bash
# Health check
curl http://localhost:8000/health

# Analyze code
curl -X POST http://localhost:8000/analyze/detailed \
  -H "Content-Type: application/json" \
  -d '{
    "code": "for i in range(10):\n    if i = 5:",
    "language": "python",
    "level": "beginner"
  }'

# Groq explanation (Phase 2)
curl -X POST http://localhost:8000/explain/quick \
  -H "Content-Type: application/json" \
  -d '{
    "code": "x = 10 / 0",
    "language": "python",
    "level": "beginner"
  }'
```

---

## 🔍 Debugging Tips

### Check API Logs
```bash
# Live logs
docker logs polymentor-api -f

# With timestamps
docker logs polymentor-api -f --timestamps

# Last N lines
docker logs polymentor-api --tail 100
```

### Debug MongoDB
```bash
# Check connection
python -c "
import pymongo
client = pymongo.MongoClient('mongodb://admin:password@localhost:27017')
print(client.server_info())
"

# Check collection size
db.user_interactions.stats()

# Find slow queries
db.setProfilingLevel(1)
db.system.profile.find().sort({ ts: -1 }).limit(5)
```

### Debug Groq API
```python
# Test API key
from groq import Groq
client = Groq()
models = client.models.list()
print(f"✓ Connected: {models.data[0].id}")

# Test explanation
response = client.chat.completions.create(
    model="llama-3.1-70b-versatile",
    messages=[{"role": "user", "content": "Say hello"}],
    max_tokens=100
)
print(response.choices[0].message.content)
```

### Check Disk Space
```bash
# Overall usage
df -h

# MongoDB specific
du -sh /var/lib/docker/volumes/mongo_data

# Models directory
du -sh models_saved/
```

---

## 🚨 Common Issues & Solutions

### MongoDB Connection Fails

```
Error: Server selection timed out after 30000ms
```

**Solution**:
```bash
# Verify container running
docker ps | grep mongo

# Check logs
docker logs polymentor-mongo

# Restart MongoDB
docker restart polymentor-mongo

# Test connection
mongosh --uri "mongodb://admin:password@localhost:27017"
```

### Groq API Rate Limited

```
Error: 429 Rate Limit Exceeded
```

**Solution**:
```python
# Implement retry with exponential backoff
from tenacity import retry, wait_exponential, stop_after_attempt

@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(3)
)
def call_groq():
    # Your code here
    pass
```

### Out of Memory During Training

```
Error: CUDA out of memory
```

**Solution**:
```bash
# Reduce batch size
export TRAINING_BATCH_SIZE=8

# Use CPU
export CUDA_VISIBLE_DEVICES=""

# Or use quantization
pip install bitsandbytes
# Use load_in_8bit=True
```

### Model Loading Too Slow

```
# Use smaller model
python -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('t5-small')  # Instead of t5-large
"

# Or cache model locally
export HF_HOME=/cache/huggingface
```

---

## 📊 Performance Benchmarks

### Target Performance

| Component | Metric | Target |
|-----------|--------|--------|
| API | Response Time (p50) | <100ms |
| API | Response Time (p95) | <500ms |
| Code Analyzer | Analysis Time | <5ms |
| Groq | Response Time | 2-10s |
| MongoDB | Query Time | <50ms |
| Model Inference | Explanation Generation | <2s |

### Benchmarking

```bash
# Load test
locust -f tests/load_test.py --host=http://localhost:8000 -u 100 -r 10

# Memory profiling
python -m memory_profiler src/api/app.py

# CPU profiling
python -m cProfile -s cumtime src/api/app.py

# Database query profiling
db.setProfilingLevel(1)
db.system.profile.find().pretty()
```

---

## 📦 Dependency Management

### Core Dependencies
```
fastapi==0.136.0
pydantic==2.5.0
tree-sitter==0.21.1
pytest==7.4.3
```

### New for v2.0
```
groq==0.4.2
pymongo==4.6.0
motor==3.3.2
mlflow==2.10.0
transformers==4.36.0
torch==2.1.0
apache-airflow==2.8.0
prometheus-client==0.19.0
```

### Installation
```bash
# Install all
pip install -r requirements.txt -r requirements-v2.txt

# Update single package
pip install --upgrade transformers

# Check versions
pip list | grep -E "(groq|mongo|mlflow|transformers)"
```

---

## 🔄 Git Workflow

### Feature Development
```bash
# Create feature branch
git checkout -b feature/mongodb-integration

# Make changes and commit
git add src/database/
git commit -m "feat: Add MongoDB integration module"

# Push to GitHub
git push origin feature/mongodb-integration

# Create Pull Request
# (on GitHub)
```

### Merging
```bash
# Update main
git checkout main
git pull origin main

# Merge feature
git merge feature/mongodb-integration

# Push
git push origin main

# Delete feature branch
git branch -d feature/mongodb-integration
git push origin --delete feature/mongodb-integration
```

---

## 📞 Useful Resources

### Documentation
- [FastAPI Docs](https://fastapi.tiangolo.com)
- [Groq API](https://console.groq.com/docs)
- [MongoDB Docs](https://docs.mongodb.com)
- [Hugging Face](https://huggingface.co/docs)
- [MLflow](https://mlflow.org/docs)

### Tools
- **Postman**: API testing - https://www.postman.com
- **MongoDB Compass**: GUI for MongoDB - https://www.mongodb.com/products/compass
- **MLflow UI**: Model tracking - http://localhost:5000

### Community
- GitHub Issues: Report bugs
- Discussions: Ask questions
- Wiki: Share knowledge

---

## 📈 Progress Tracking

### Implementation Status

- [x] Phase 1: Documentation & Planning
- [ ] Phase 2: MongoDB Integration (Week 1)
- [ ] Phase 3: Groq API (Week 1-2)
- [ ] Phase 4: MLOps Pipeline (Week 2-3)
- [ ] Phase 5: Hugging Face (Week 3)
- [ ] Phase 6: Deployment & Monitoring (Week 4)

### Next Steps
1. Review documentation
2. Setup development environment
3. Start Phase 1 (MongoDB)
4. Create feature branches
5. Test locally before pushing

---

**Version**: 2.0  
**Last Updated**: June 11, 2026  
**For Questions**: See COMPLETE_PROJECT_DOCUMENTATION.md
