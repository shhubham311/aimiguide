# Deployment Types: On-Device & On-Server

## Why Deployment Matters for ML/AI

Training a model is only 10% of the work—the other 90% is **getting it into production** where it delivers real value. Deployment decisions affect latency, cost, scalability, privacy, and user experience. Choosing between **on-device (edge)** and **on-server (cloud)** deployment determines whether your model runs locally on the user's phone or remotely on a cloud server.

This lesson covers both deployment paradigms, their tradeoffs, optimization techniques, and when to choose each.

---

## The Deployment Decision Matrix

```
┌────────────────────────────────────────────────────────────────┐
│                    Deployment Decision                          │
│                                                                 │
│                    Privacy needed?                              │
│                    ┌─────────┐                                  │
│                   YES       NO                                 │
│                  ┌──┐      ┌──┐                                │
│                  │ON│      │  │                                │
│                  │  │      │  │  Low latency                   │
│                  │DEV│      │  │  needed?                      │
│                  │ICE│      │  │  ┌────┐┌────┐                 │
│                  └──┘      │YES│  │ON  │SERV│                 │
│                            │   │  │DEV │ER  │                 │
│                            └──┘  └────┘└────┘                 │
│                                                                 │
│  On-Device:     On-Server:                                      │
│  • Offline      • Scalable                                      │
│  • Private      • Easy updates                                  │
│  • Fast         • Large models                                  │
│  • Limited HW   • GPU access                                    │
└────────────────────────────────────────────────────────────────┘
```

---

## On-Server Deployment

### Architecture

```
┌──────────┐    HTTP/gRPC     ┌──────────────┐    Inference     ┌──────────────┐
│  Client   │ ◄─────────────► │  API Server   │ ──────────────► │  ML Model    │
│  (Mobile, │   Request/      │  (FastAPI,    │  model.predict() │  (GPU/CPU)   │
│   Web,    │   Response      │  Flask, TFS) │                 │              │
│   IoT)    │                 │              │                 └──────────────┘
└──────────┘                  └──────┬───────┘
                                     │
                              ┌──────▼───────┐
                              │  Load Balancer│
                              │  (ALB/Nginx)  │
                              └──────┬───────┘
                                     │
                    ┌────────────────┼────────────────┐
                    ▼                ▼                ▼
              ┌──────────┐   ┌──────────┐    ┌──────────┐
              │ Server 1 │   │ Server 2 │    │ Server N │
              │ GPU: A100│   │ GPU: A100│    │ GPU: T4  │
              │ Model v2 │   │ Model v2 │    │ Model v2 │
              └──────────┘   └──────────┘    └──────────┘
```

### Serving Frameworks

```python
# TorchServe (PyTorch's official model server)
# pip install torchserve torch-model-archiver torch-workflow-archiver

# Create a handler.py
class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None

    def initialize(self, context):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        model_dir = context.system_properties.get("model_dir")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model.eval()

    def handle(self, data, context):
        import torch
        text = data[0].get("body").decode('utf-8')
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits).item()
        return [{"prediction": prediction}]

# Package model
# torch-model-archiver --model-name sentiment \
#   --version 1.0 --handler handler.py \
#   --serialized-file model.pt

# Start server
# torchserve --start --model-store model_store --models sentiment.mar
```

### Framework Comparison

| Framework | Best For | Pros | Cons |
|-----------|----------|------|------|
| **TorchServe** | PyTorch models | Official, multi-model | Steeper learning curve |
| **TF Serving** | TensorFlow | Production-grade | TF-specific |
| **Triton** | Multi-framework | GPU optimization | Complex setup |
| **BentoML** | General Python | Easy, flexible | Less GPU optimization |
| **Ray Serve** | Distributed | Scalable, composable | Overhead for simple cases |

---

## On-Device Deployment

### Architecture

```
┌─────────────────────────────────────────┐
│              Mobile Device               │
│                                          │
│  ┌──────────────┐                        │
│  │   App UI      │                        │
│  │  (Camera,     │                        │
│  │   Text Input) │                        │
│  └──────┬───────┘                        │
│         │                                │
│  ┌──────▼───────┐                        │
│  │  Preprocessing│                        │
│  │  (Resize,     │                        │
│  │   Normalize)  │                        │
│  └──────┬───────┘                        │
│         │                                │
│  ┌──────▼───────┐    ┌──────────────┐   │
│  │  ML Model     │    │  Model Size  │   │
│  │  (TFLite,     │    │  ───────────  │   │
│  │   CoreML,     │    │  < 10MB ideal│   │
│  │   ONNX)       │    │  < 50MB ok   │   │
│  └──────┬───────┘    └──────────────┘   │
│         │                                │
│  ┌──────▼───────┐                        │
│  │  Postprocess  │                        │
│  │  (NMS,        │                        │
│  │   Threshold)  │                        │
│  └──────┬───────┘                        │
│         │                                │
│  ┌──────▼───────┐                        │
│  │  App Display  │                        │
│  └──────────────┘                        │
│                                          │
│  Hardware: CPU, NPU, DSP                 │
│  No internet needed!                     │
└─────────────────────────────────────────┘
```

### Model Conversion Pipeline

```
┌────────────┐     Convert      ┌────────────┐     Optimize     ┌────────────┐
│  PyTorch/  │ ──────────────► │  TFLite    │ ──────────────► │  Quantized │
│  TF Model  │  (float32)      │  (float32) │  (int8)         │  TFLite    │
│  (768MB)   │                  │  (50MB)    │                  │  (12MB)    │
└────────────┘                  └────────────┘                  └────────────┘
```

### PyTorch to TFLite

```python
import torch
import torch.nn as nn
import numpy as np

# 1. Define and train model (or load pre-trained)
class SentimentModel(nn.Module):
    def __init__(self, vocab_size=30000, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        return self.fc(hidden[-1])

model = SentimentModel()
model.eval()

# 2. Export to ONNX
dummy_input = torch.randint(0, 30000, (1, 100))
torch.onnx.export(model, dummy_input, "sentiment.onnx",
                  input_names=['input_ids'],
                  output_names=['logits'],
                  dynamic_axes={'input_ids': {0: 'batch', 1: 'seq'}})

# 3. Convert ONNX to TFLite
# Using onnx-tf and tf.lite
import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("sentiment.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph("sentiment_tf")

# Convert to TFLite
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model("sentiment_tf")
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # int8 quantization
tflite_model = converter.convert()

with open("sentiment_quantized.tflite", "wb") as f:
    f.write(tflite_model)

print(f"Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
```

### On-Device Inference (Android/iOS)

```python
# Python simulation of on-device inference
import numpy as np
import tflite_runtime.interpreter as tflite  # For Raspberry Pi / edge devices

interpreter = tflite.Interpreter(model_path="sentiment_quantized.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(text_tokens):
    # text_tokens: list of token IDs, padded/truncated to 100
    input_data = np.array([text_tokens], dtype=np.int32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    confidence = np.max(output_data) / np.sum(output_data)
    return prediction, confidence

# Usage
tokens = [12, 345, 67, 890, ...]  # Tokenized text (length 100)
label, conf = predict(tokens)
print(f"Prediction: {'positive' if label == 1 else 'negative'}, Confidence: {conf:.2f}")
```

---

## Model Quantization

### Types of Quantization

```
┌──────────────────────────────────────────────────────────────┐
│                Quantization Tradeoffs                         │
│                                                               │
│  Type           Size    Speed    Accuracy    When to Use      │
│  ─────────────────────────────────────────────────────────   │
│  FP32 (full)    100%    1×       100%        Development     │
│  FP16           50%     2×       ~100%       GPU serving     │
│  INT8           25%     4×       -0.5~2%     On-device       │
│  INT4           12.5%   6~8×     -1~5%       Edge, extreme   │
│  Binary (1-bit) ~3%    10×+     -5~15%      Research        │
│                                                               │
│  Accuracy loss depends on:                                   │
│  - Model architecture                                        │
│  - Dataset complexity                                        │
│  - Quantization method (post-training vs. QAT)               │
└──────────────────────────────────────────────────────────────┘
```

### Post-Training Quantization (PTQ)

```python
import torch
from torch.quantization import quantize_dynamic

# Dynamic quantization: quantizes weights only, activations stay float
quantized_model = quantize_dynamic(
    model,                        # Original model
    {nn.Linear, nn.LSTM},        # Layers to quantize
    dtype=torch.qint8            # Target dtype
)

# Save
torch.save(quantized_model.state_dict(), "model_quantized.pt")
print(f"Original: {sum(p.numel() for p in model.parameters()) * 4 / 1024**2:.1f} MB")
print(f"Quantized: {sum(p.numel() for p in quantized_model.parameters()) * 1 / 1024**2:.1f} MB")
```

### Quantization-Aware Training (QAT)

```python
import torch
from torch.quantization import prepare_qat, convert

# Insert fake-quantization modules
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
model_prepared = prepare_qat(model)

# Fine-tune with fake quantization
for epoch in range(3):
    for batch in train_loader:
        output = model_prepared(batch)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

# Convert to actual quantized model
model_quantized = convert(model_prepared.eval())
```

---

## Comparison: On-Device vs On-Server

```
┌───────────────────┬──────────────────┬──────────────────┐
│ Feature           │ On-Device        │ On-Server       │
├───────────────────┼──────────────────┼──────────────────┤
│ Latency           │ < 10ms           │ 50-500ms         │
│ Privacy           │ Full (data stays)│ Data sent to     │
│ Offline           │ Yes              │ No               │
│ Model size limit  │ 5-50MB           │ Unlimited        │
│ GPU available     │ Rarely           │ Yes (A100, etc.) │
│ Updates           │ App store update │ Instant          │
│ Cost per request  │ Zero (after DL)  │ $0.001-0.10      │
│ Scalability       │ Per-device       │ Elastic          │
│ Use cases         │ Camera, voice,   │ LLMs, large     │
│                   │ keyboard, AR     │ models, search   │
└───────────────────┴──────────────────┴──────────────────┘
```

---

## Exercises

### Exercise 1: Choose Deployment Strategy
For each scenario, recommend on-device or on-server deployment with reasoning:
a) Voice assistant on a smart speaker
b) Medical image analysis for radiologists
c) Face unlock on a phone
d) ChatGPT-style conversational AI

**Solution:**
a) **On-device**: Privacy (voice data shouldn't leave home), low latency (real-time), small model (keyword spotting + wake word detection)
b) **On-server**: Large model needed (3D CNN), high accuracy critical, cloud GPU available, radiologist has internet
c) **On-device**: Privacy (biometric data), < 100ms latency, security (data never leaves device)
d) **On-server**: Large LLM (7B+ params), needs GPU, frequent updates, acceptable latency (~1s)

### Exercise 2: Calculate Quantized Size
A model has 10M parameters. What's its memory footprint in FP32, FP16, and INT8?

**Solution:**
```
FP32: 10M × 4 bytes  = 40 MB
FP16: 10M × 2 bytes  = 20 MB
INT8: 10M × 1 byte   = 10 MB

INT8 is 4× smaller than FP32 with typically < 2% accuracy loss.
```

### Exercise 3: Design a Hybrid Strategy
Design a deployment strategy for a mobile app that needs both fast local predictions and high-quality cloud predictions.

**Solution:**
```python
class HybridPredictor:
    def __init__(self):
        self.local_model = load_tflite_model("small_model.tflite")   # 5MB
        self.server_url = "https://api.example.com/predict"

    def predict(self, input_data, confidence_threshold=0.8):
        # Try local first (fast, offline)
        local_pred, local_conf = self.local_model.predict(input_data)

        if local_conf >= confidence_threshold:
            return {"source": "local", "prediction": local_pred}

        # Fall back to server (slower, more accurate)
        server_response = requests.post(self.server_url, json=input_data)
        return {"source": "server", "prediction": server_response.json()}
```

---

## Key Takeaways

1. **On-server** deployment offers unlimited model size, easy updates, and GPU access—best for large models
2. **On-device** deployment offers privacy, offline capability, and low latency—best for mobile/edge
3. **Quantization** reduces model size 4× (INT8) with minimal accuracy loss
4. **Hybrid strategies** combine local speed with cloud accuracy via confidence-based fallback
5. **Choose based on**: latency requirements, privacy needs, model size, offline requirements, update frequency
