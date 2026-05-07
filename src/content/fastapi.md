# API Development Using FastAPI

## Why FastAPI Matters for ML/AI

Once you train a model, you need to **serve it to users**. Whether it's a web app calling your sentiment analyzer, a mobile app using your image classifier, or an internal tool querying your recommendation engine—**APIs are how ML models reach the real world**.

FastAPI is the modern Python web framework purpose-built for APIs. It's **fast** (built on Starlette + Uvicorn), **type-safe** (leverages Python type hints), **auto-documented** (generates OpenAPI/Swagger docs), and **perfect for ML serving** (native async, Pydantic validation, easy dependency injection).

---

## Core Concepts

### Request-Response Cycle

```
┌──────────┐     HTTP Request      ┌──────────────┐     Inference      ┌──────────────┐
│  Client   │ ──────────────────►  │   FastAPI     │ ───────────────►  │  ML Model    │
│  (App,    │  GET /predict        │   Server      │  model.predict()  │  (.pt, .pkl) │
│   curl,   │  {text: "great!"}    │               │                   │              │
│   browser)│                      │  Routes       │                   └──────┬───────┘
│           │ ◄──────────────────  │  Middleware   │ ◄─────────────────        │
│           │  JSON Response       │  Validation   │  {prediction: "pos"}     │
│           │  {"sentiment":       │  Auth         │                          │
│           │   "positive",        │  Errors       │                          │
│           │   "confidence": 0.95}│               │                          │
└──────────┘                      └──────────────┘                          │
                                                                            │
```

### Minimal FastAPI App

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="ML Prediction API", version="1.0")

class PredictionRequest(BaseModel):
    text: str
    model_version: str = "v1"

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    model_version: str

@app.get("/")
def health_check():
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Simulated prediction
    confidence = 0.95 if "great" in request.text.lower() else 0.3
    sentiment = "positive" if confidence > 0.5 else "negative"

    return PredictionResponse(
        text=request.text,
        sentiment=sentiment,
        confidence=confidence,
        model_version=request.model_version
    )
```

Run with: `uvicorn main:app --reload --port 8000`

Auto-generated docs at: `http://localhost:8000/docs` (Swagger UI)

---

## Building a Complete ML API

### Project Structure

```
ml_api/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI app entry point
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py       # Pydantic models
│   │   └── ml_model.py      # Model loading & inference
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── predictions.py   # Prediction endpoints
│   │   └── health.py        # Health check endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   └── preprocessor.py  # Text preprocessing
│   └── core/
│       ├── __init__.py
│       └── config.py        # Settings/configuration
├── requirements.txt
├── Dockerfile
└── tests/
    └── test_api.py
```

### Pydantic Schemas (Validation)

```python
# app/models/schemas.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from enum import Enum

class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class TextInput(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000,
                       description="Input text for classification")
    model_version: str = Field(default="v1", pattern=r"^v\d+$")

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be whitespace only")
        return v.strip()

class PredictionOutput(BaseModel):
    text: str
    sentiment: Sentiment
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str
    processing_time_ms: float

class BatchInput(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model_version: str = "v1"

class BatchOutput(BaseModel):
    predictions: List[PredictionOutput]
    total_processed: int
```

### Model Service

```python
# app/models/ml_model.py
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import time
from functools import lru_cache

class SentimentModel:
    def __init__(self, model_name="distilbert-base-uncased-finetuned-sst-2-english"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.labels = {0: "negative", 1: "positive"}

    def predict(self, text: str) -> dict:
        start = time.time()

        inputs = self.tokenizer(text, return_tensors="pt",
                                truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred_class = torch.argmax(probs, dim=-1).item()
            confidence = probs[0, pred_class].item()

        processing_time = (time.time() - start) * 1000

        return {
            "sentiment": self.labels[pred_class],
            "confidence": round(confidence, 4),
            "processing_time_ms": round(processing_time, 2)
        }

# Singleton: load model once
_model_instance = None

def get_model():
    global _model_instance
    if _model_instance is None:
        _model_instance = SentimentModel()
    return _model_instance
```

### Routers

```python
# app/routers/predictions.py
from fastapi import APIRouter, Depends, HTTPException
from ..models.schemas import TextInput, PredictionOutput, BatchInput, BatchOutput
from ..models.ml_model import get_model

router = APIRouter(prefix="/api/v1", tags=["predictions"])

@router.post("/predict", response_model=PredictionOutput)
async def predict_sentiment(request: TextInput):
    model = get_model()
    result = model.predict(request.text)

    return PredictionOutput(
        text=request.text,
        sentiment=result["sentiment"],
        confidence=result["confidence"],
        model_version=request.model_version,
        processing_time_ms=result["processing_time_ms"]
    )

@router.post("/predict/batch", response_model=BatchOutput)
async def predict_batch(request: BatchInput):
    model = get_model()
    predictions = []

    for text in request.texts:
        result = model.predict(text)
        predictions.append(PredictionOutput(
            text=text,
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            model_version=request.model_version,
            processing_time_ms=result["processing_time_ms"]
        ))

    return BatchOutput(
        predictions=predictions,
        total_processed=len(predictions)
    )
```

### Main App with Middleware

```python
# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import logging

from .routers import predictions, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready ML API for text sentiment classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = (time.time() - start_time) * 1000
    response.headers["X-Process-Time"] = str(round(process_time, 2))
    logger.info(f"{request.method} {request.url} → {response.status_code} ({process_time:.2f}ms)")
    return response

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

# Include routers
app.include_router(predictions.router)
app.include_router(health.router, tags=["health"])

@app.on_event("startup")
async def startup_event():
    logger.info("Loading ML model...")
    get_model()  # Pre-load model
    logger.info("Model loaded successfully!")
```

---

## Authentication & Rate Limiting

```python
from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader
import secrets

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME)

# In production, store in database or environment variable
API_KEYS = {"sk-ml-api-12345": "production"}

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return api_key

# Use in routes
@router.post("/predict")
async def predict(request: TextInput, api_key: str = Depends(verify_api_key)):
    ...
```

---

## Dockerizing the API

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
# Build and run
docker build -t ml-api .
docker run -p 8000:8000 -e API_KEY=sk-ml-api-12345 ml-api
```

---

## Exercises

### Exercise 1: Add Input Validation
Create an endpoint `/classify` that accepts an image file upload and validates that it's a valid image format (jpg, png).

**Solution:**
```python
from fastapi import UploadFile, File
from PIL import Image
import io

@router.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(400, "Only JPG and PNG images allowed")

    # Validate image is readable
    try:
        image = Image.open(io.BytesIO(await file.read()))
        image.verify()
    except Exception:
        raise HTTPException(400, "Invalid image file")

    return {"filename": file.filename, "size": image.size, "status": "valid"}
```

### Exercise 2: Async Batch Processing
How would you handle 1000 prediction requests efficiently?

**Solution:** Use async with a task queue:
```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

executor = ThreadPoolExecutor(max_workers=4)

@router.post("/predict/async-batch")
async def async_batch(request: BatchInput):
    loop = asyncio.get_event_loop()
    model = get_model()

    tasks = [
        loop.run_in_executor(executor, model.predict, text)
        for text in request.texts
    ]
    results = await asyncio.gather(*tasks)

    return {"results": results, "count": len(results)}
```

### Exercise 3: Add Health Check with Model Status
Create a health endpoint that checks model availability.

**Solution:**
```python
@router.get("/health")
async def health_check():
    try:
        model = get_model()
        return {
            "status": "healthy",
            "model_loaded": model.model is not None,
            "model_device": str(next(model.model.parameters()).device),
            "version": "1.0.0"
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```

---

## Key Takeaways

1. **FastAPI** provides type-safe, auto-documented, high-performance APIs for ML model serving
2. **Pydantic** validates all inputs/outputs automatically—prevents bad data reaching your model
3. **Dependency injection** (`Depends`) manages model lifecycle, auth, and shared resources
4. **Middleware** adds cross-cutting concerns: CORS, timing, logging, error handling
5. **Docker** ensures consistent deployment across development and production environments
