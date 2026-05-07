# Testing Model Performance Using API Locally

## Why Testing Matters for ML/AI

Deploying an ML model without testing is like launching a rocket without quality assurance. **API-level testing** verifies that your model server works correctly end-to-end: inputs are validated, predictions are accurate, latency is acceptable, errors are handled gracefully, and the system degrades predictably under load.

Testing ML APIs is different from traditional software testing because:
- **Model outputs are probabilistic** (confidence scores, probabilities)
- **Input space is vast** (any text, any image)
- **Performance varies** with hardware, batch size, and model complexity
- **Edge cases matter** (empty inputs, malformed data, adversarial inputs)

This lesson covers unit tests, integration tests, load tests, and monitoring for ML APIs.

---

## Testing Pyramid for ML APIs

```
                    ╱╲
                   ╱  ╲          E2E Tests
                  ╱    ╲         (Full API calls,
                 ╱──────╲         real model inference)
                ╱        ╲
               ╱  Integration ╲   Integration Tests
              ╱      Tests      ╲  (API routes, DB,
             ╱────────────────────╲  preprocessing)
            ╱                      ╲
           ╱     Unit Tests         ╲  Unit Tests
          ╱    (Functions, classes,   ╲ (Individual functions,
         ╱       model components)     ╲  preprocessing, math)
        ╱────────────────────────────────╲

  More tests          Fewer tests
  Faster to run       Slower to run
  Cheaper to write    More expensive
```

---

## Unit Testing with pytest

### Testing Preprocessing

```python
# app/services/preprocessor.py
import re
from typing import List

def clean_text(text: str) -> str:
    """Clean and normalize input text."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)           # Normalize whitespace
    text = re.sub(r'[^\w\s.,!?]', '', text)    # Remove special chars
    text = text.strip()
    return text

def tokenize_text(text: str, max_tokens: int = 512) -> List[str]:
    """Simple whitespace tokenizer with truncation."""
    tokens = text.split()
    return tokens[:max_tokens]

def compute_confidence(logits: List[float]) -> dict:
    """Convert logits to probabilities and return prediction."""
    import math
    max_logit = max(logits)
    exp_logits = [math.exp(l - max_logit) for l in logits]  # Numerical stability
    sum_exp = sum(exp_logits)
    probs = [e / sum_exp for e in exp_logits]
    pred_idx = probs.index(max(probs))
    return {
        "prediction": pred_idx,
        "confidence": max(probs),
        "probabilities": probs
    }
```

### Unit Tests

```python
# tests/test_preprocessor.py
import pytest
from app.services.preprocessor import clean_text, tokenize_text, compute_confidence

class TestCleanText:
    def test_basic_cleaning(self):
        assert clean_text("Hello  World!") == "hello world"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_none_input(self):
        assert clean_text(None) == ""

    def test_special_characters(self):
        result = clean_text("Price: $100!!! #deal")
        assert "$" not in result
        assert "#" not in result

    def test_multiple_spaces(self):
        assert clean_text("Hello    World") == "hello world"

    def test_preserve_punctuation(self):
        assert clean_text("Great product!") == "great product!"

class TestTokenizeText:
    def test_basic_tokenize(self):
        assert tokenize_text("hello world") == ["hello", "world"]

    def test_truncation(self):
        tokens = tokenize_text(" ".join(["word"] * 1000), max_tokens=10)
        assert len(tokens) == 10

    def test_empty_input(self):
        assert tokenize_text("") == []

class TestComputeConfidence:
    def test_equal_logits(self):
        result = compute_confidence([0.0, 0.0, 0.0])
        assert result["confidence"] == pytest.approx(1/3, abs=0.01)

    def test_one_dominant(self):
        result = compute_confidence([10.0, 0.0, 0.0])
        assert result["prediction"] == 0
        assert result["confidence"] > 0.99

    def test_probabilities_sum_to_one(self):
        result = compute_confidence([1.5, 0.5, 0.3])
        assert sum(result["probabilities"]) == pytest.approx(1.0, abs=0.001)
```

---

## Integration Testing the API

### Test Client with FastAPI

```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

class TestHealthEndpoint:
    def test_health_check(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

class TestPredictionEndpoint:
    def test_positive_sentiment(self):
        response = client.post("/api/v1/predict", json={
            "text": "This is absolutely wonderful!",
            "model_version": "v1"
        })
        assert response.status_code == 200
        data = response.json()
        assert "sentiment" in data
        assert "confidence" in data
        assert 0.0 <= data["confidence"] <= 1.0

    def test_negative_sentiment(self):
        response = client.post("/api/v1/predict", json={
            "text": "Terrible experience, never again.",
            "model_version": "v1"
        })
        assert response.status_code == 200

    def test_empty_text_rejected(self):
        response = client.post("/api/v1/predict", json={
            "text": "",
            "model_version": "v1"
        })
        assert response.status_code == 422  # Validation error

    def test_missing_field(self):
        response = client.post("/api/v1/predict", json={
            "model_version": "v1"
        })
        assert response.status_code == 422

    def test_long_text_handled(self):
        long_text = "Great! " * 5000  # 10,000+ chars
        response = client.post("/api/v1/predict", json={
            "text": long_text,
            "model_version": "v1"
        })
        assert response.status_code == 200

class TestBatchEndpoint:
    def test_batch_prediction(self):
        response = client.post("/api/v1/predict/batch", json={
            "texts": ["Great product!", "Terrible service", "Okay experience"],
            "model_version": "v1"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["total_processed"] == 3
        assert len(data["predictions"]) == 3

    def test_empty_batch_rejected(self):
        response = client.post("/api/v1/predict/batch", json={
            "texts": [],
            "model_version": "v1"
        })
        assert response.status_code == 422

    def test_batch_exceeds_limit(self):
        response = client.post("/api/v1/predict/batch", json={
            "texts": ["text"] * 101,  # Max is 100
            "model_version": "v1"
        })
        assert response.status_code == 422
```

### Testing with Mocked Model

```python
from unittest.mock import patch, MagicMock

class TestWithMockModel:
    @patch("app.models.ml_model.get_model")
    def test_mocked_prediction(self, mock_get_model):
        # Setup mock
        mock_model = MagicMock()
        mock_model.predict.return_value = {
            "sentiment": "positive",
            "confidence": 0.99,
            "processing_time_ms": 5.0
        }
        mock_get_model.return_value = mock_model

        # Test
        response = client.post("/api/v1/predict", json={
            "text": "any text",
            "model_version": "v1"
        })
        assert response.status_code == 200
        assert response.json()["sentiment"] == "positive"
        mock_model.predict.assert_called_once_with("any text")
```

---

## Load Testing

### Using Locust

```python
# locustfile.py
from locust import HttpUser, task, between

class MLApiUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)  # Weight: 3x more frequent
    def predict_single(self):
        self.client.post("/api/v1/predict", json={
            "text": "This product is amazing and I love it!",
            "model_version": "v1"
        })

    @task(1)  # Weight: 1x less frequent
    def predict_batch(self):
        self.client.post("/api/v1/predict/batch", json={
            "texts": ["Great!", "Terrible!", "Okay"],
            "model_version": "v1"
        })

    @task(1)
    def health_check(self):
        self.client.get("/health")
```

```bash
# Run load test
locust -f locustfile.py --host=http://localhost:8000

# Or headless mode
locust -f locustfile.py --host=http://localhost:8000 \
  --users 100 --spawn-rate 10 --run-time 60s --headless
```

### Load Test Metrics to Monitor

```
┌──────────────────────────────────────────────┐
│           Load Test Dashboard                 │
│                                                │
│  Metric              Target     Warning       │
│  ─────────────────────────────────────────    │
│  Median Latency      < 100ms    < 500ms      │
│  95th Percentile      < 500ms    < 2000ms     │
│  99th Percentile      < 1000ms   < 5000ms     │
│  Requests/sec        > 100      < 50         │
│  Error Rate           < 1%       < 5%         │
│  CPU Utilization     < 70%      < 90%         │
│  Memory Usage        < 80%      < 95%         │
└──────────────────────────────────────────────┘
```

---

## Testing Model Accuracy via API

### Regression Test Suite

```python
# tests/test_model_accuracy.py
import pytest

# Golden dataset: inputs with known expected outputs
GOLDEN_TESTS = [
    {"text": "I love this product!", "expected": "positive"},
    {"text": "This is the worst experience ever", "expected": "negative"},
    {"text": "It's okay, nothing special", "expected": "negative"},  # SST-2 is binary
    {"text": "Absolutely fantastic work", "expected": "positive"},
    {"text": "I would not recommend this", "expected": "negative"},
]

class TestModelAccuracy:
    def test_golden_set_accuracy(self):
        """Model should achieve > 90% accuracy on golden test set."""
        correct = 0
        for test in GOLDEN_TESTS:
            response = client.post("/api/v1/predict", json={
                "text": test["text"],
                "model_version": "v1"
            })
            assert response.status_code == 200
            if response.json()["sentiment"] == test["expected"]:
                correct += 1

        accuracy = correct / len(GOLDEN_TESTS)
        assert accuracy >= 0.9, f"Accuracy {accuracy:.2%} below 90% threshold"

    def test_confidence_calibration(self):
        """Confidence scores should be reasonable."""
        response = client.post("/api/v1/predict", json={
            "text": "This is clearly the best thing ever!!!",
            "model_version": "v1"
        })
        confidence = response.json()["confidence"]
        assert confidence > 0.8, f"Confidence too low for clearly positive text: {confidence}"

    def test_consistency(self):
        """Same input should give same output."""
        payload = {"text": "Consistent test sentence", "model_version": "v1"}
        response1 = client.post("/api/v1/predict", json=payload).json()
        response2 = client.post("/api/v1/predict", json=payload).json()
        assert response1["sentiment"] == response2["sentiment"]
```

---

## cURL Quick Reference for Manual Testing

```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Great product!", "model_version": "v1"}'

# With API key
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: sk-ml-api-12345" \
  -d '{"text": "Great product!", "model_version": "v1"}'

# Batch prediction
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Love it!", "Hate it!"], "model_version": "v1"}'

# Load test with concurrent requests
for i in {1..100}; do
  curl -s -o /dev/null -w "%{time_total}\n" \
    -X POST http://localhost:8000/api/v1/predict \
    -H "Content-Type: application/json" \
    -d '{"text": "test", "model_version": "v1"}' &
done
wait
```

---

## Exercises

### Exercise 1: Write a Test for Adversarial Input
Write a test that verifies the API handles injection attacks safely.

**Solution:**
```python
class TestSecurity:
    def test_sql_injection(self):
        response = client.post("/api/v1/predict", json={
            "text": "'; DROP TABLE users; --",
            "model_version": "v1"
        })
        # Should process as normal text, not crash
        assert response.status_code == 200
        assert "sentiment" in response.json()

    def test_extremely_long_input(self):
        response = client.post("/api/v1/predict", json={
            "text": "x" * 1_000_000,
            "model_version": "v1"
        })
        # Should either process or reject gracefully
        assert response.status_code in [200, 422]
```

### Exercise 2: Performance Benchmark
Write a test that verifies prediction latency is under 200ms.

**Solution:**
```python
import time

class TestPerformance:
    def test_latency_under_200ms(self):
        times = []
        for _ in range(20):
            start = time.time()
            response = client.post("/api/v1/predict", json={
                "text": "This is a test sentence for latency benchmarking.",
                "model_version": "v1"
            })
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
            assert response.status_code == 200

        p95 = sorted(times)[int(0.95 * len(times))]
        avg = sum(times) / len(times)
        assert avg < 200, f"Avg latency {avg:.1f}ms exceeds 200ms threshold"
        print(f"Avg: {avg:.1f}ms, P95: {p95:.1f}ms")
```

### Exercise 3: Model Versioning Test
Write a test that verifies different model versions give valid responses.

**Solution:**
```python
class TestModelVersioning:
    @pytest.mark.parametrize("version", ["v1", "v2"])
    def test_valid_version(self, version):
        response = client.post("/api/v1/predict", json={
            "text": "Test input",
            "model_version": version
        })
        assert response.status_code == 200
        assert response.json()["model_version"] == version

    def test_invalid_version_format(self):
        response = client.post("/api/v1/predict", json={
            "text": "Test",
            "model_version": "invalid"
        })
        assert response.status_code == 422
```

---

## Key Takeaways

1. **Test pyramid**: Many fast unit tests → fewer integration tests → few slow E2E tests
2. **Golden test sets** catch model regressions by comparing against known-good outputs
3. **Load testing** (Locust) validates latency, throughput, and error rates under realistic traffic
4. **Mock models** isolate API logic from model inference for faster, deterministic tests
5. **Security testing** ensures your API handles adversarial and malformed inputs gracefully
