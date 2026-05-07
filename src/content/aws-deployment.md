# Model Deployment on AWS: SageMaker & EC2

## Why AWS Deployment Matters

Amazon Web Services (AWS) is the dominant cloud platform for ML deployment, holding ~32% market share. **Amazon SageMaker** provides a fully managed ML platform with built-in training, tuning, and deployment. **EC2** (Elastic Compute Cloud) offers full control over infrastructure for custom deployments. Together, they cover the spectrum from "just upload your model" to "build everything from scratch."

Understanding AWS deployment is critical because most enterprise ML systems run on cloud infrastructure, and SageMaker skills are consistently among the most requested in ML engineering job postings.

---

## AWS ML Deployment Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                   AWS ML Deployment Options                       │
│                                                                    │
│  Managed (Easy)              Self-Managed (Control)                │
│  ──────────────────          ───────────────────────               │
│                                                                    │
│  ┌───────────────┐          ┌───────────────┐                     │
│  │  SageMaker    │          │  EC2 +        │                     │
│  │  Real-time    │          │  FastAPI      │                     │
│  │  Endpoints    │          │  + Nginx      │                     │
│  │  (1-click)    │          │  + Docker     │                     │
│  └───────────────┘          └───────────────┘                     │
│                                                                    │
│  ┌───────────────┐          ┌───────────────┐                     │
│  │  SageMaker    │          │  ECS/EKS      │                     │
│  │  Serverless   │          │  (Container)  │                     │
│  │  (auto-scale) │          │               │                     │
│  └───────────────┘          └───────────────┘                     │
│                                                                    │
│  ┌───────────────┐          ┌───────────────┐                     │
│  │  SageMaker    │          │  Lambda +     │                     │
│  │  Batch        │          │  API Gateway  │                     │
│  │  Transform   │          │  (lightweight)│                     │
│  └───────────────┘          └───────────────┘                     │
│                                                                    │
│  ┌───────────────┐                                               │
│  │  SageMaker    │                                               │
│  │  Asynchronous │                                               │
│  │  Endpoints    │                                               │
│  └───────────────┘                                               │
└──────────────────────────────────────────────────────────────────┘
```

---

## SageMaker: Fully Managed Deployment

### SageMaker Endpoint Architecture

```
┌────────────┐       ┌──────────────────┐       ┌──────────────┐
│  Client    │──────►│  SageMaker       │──────►│  Model       │
│  (Invoke   │       │  Endpoint        │       │  Container   │
│  Endpoint) │◄──────│  (Load Balanced) │◄──────│  (Docker)    │
│            │       │                  │       │              │
│  SDK/API   │       │  ┌───┐ ┌───┐ ┌───┐      │  model.pt    │
│            │       │  │ML │ │ML │ │ML │      │  handler.py  │
│            │       │  │I1 │ │I2 │ │I3 │      │  serve       │
│            │       │  └───┘ └───┘ └───┘      └──────────────┘
└────────────┘       └──────────────────┘
                           │
                      Auto-scaling
                      based on:
                      - Invocations/sec
                      - CPU utilization
```

### Deploying to SageMaker with Python SDK

```python
import sagemaker
from sagemaker.pytorch import PyTorchModel
from sagemaker.model_monitor import DataCaptureConfig

# Initialize SageMaker session
role = "arn:aws:iam::123456789:sagemaker-execution-role"
sagemaker_session = sagemaker.Session()

# Option 1: Deploy from local model artifacts
pytorch_model = PyTorchModel(
    model_data="s3://my-bucket/model.tar.gz",
    role=role,
    framework_version="2.0",
    py_version="py310",
    entry_point="inference.py",      # Your inference code
    source_dir="src/",               # Additional code files
    sagemaker_session=sagemaker_session,
)

# Deploy to real-time endpoint
predictor = pytorch_model.deploy(
    initial_instance_count=2,         # Number of instances
    instance_type="ml.m5.xlarge",     # Instance type
    endpoint_name="sentiment-endpoint",
    data_capture_config=DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,       # Log all requests
        destination_s3_uri="s3://my-bucket/capture/"
    ),
)

# Make predictions
response = predictor.predict({
    "inputs": "This movie was fantastic!"
})
print(response)  # {"prediction": "positive", "confidence": 0.95}

# Clean up
predictor.delete_endpoint()
```

### SageMaker Inference Handler

```python
# inference.py — deployed inside the SageMaker container
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ModelHandler:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def initialize(self, context):
        """Called once when the endpoint is created."""
        model_dir = context.model_dir  # S3 model artifacts extracted here
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.labels = {0: "negative", 1: "positive"}

    def preprocess(self, request):
        """Parse and tokenize incoming request."""
        text = request["inputs"]
        inputs = self.tokenizer(
            text, return_tensors="pt",
            truncation=True, max_length=512,
            padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def inference(self, inputs):
        """Run model inference."""
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        return probs

    def postprocess(self, probs):
        """Convert model output to API response."""
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0, pred_idx].item()
        return {
            "prediction": self.labels[pred_idx],
            "confidence": round(confidence, 4)
        }

# Simplified handler functions (alternative approach)
model_handler = ModelHandler()

def model_fn(model_dir):
    """Load model from S3 artifacts."""
    model_handler.initialize(type('Context', (), {'model_dir': model_dir})())
    return model_handler

def transform_fn(model, request_body, content_type, accept_type):
    """Transform request → inference → response."""
    data = json.loads(request_body)
    inputs = model.preprocess(data)
    probs = model.inference(inputs)
    result = model.postprocess(probs)
    return json.dumps(result), accept_type
```

### SageMaker Auto-Scaling

```python
import boto3

sm_client = boto3.client("application-autoscaling")

# Register scalable target
sm_client.register_scalable_target(
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/sentiment-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    MinCapacity=1,
    MaxCapacity=10,
)

# Add scaling policy based on invocations
sm_client.put_scaling_policy(
    PolicyName="SageMakerEndpointScaling",
    ServiceNamespace="sagemaker",
    ResourceId="endpoint/sentiment-endpoint/variant/AllTraffic",
    ScalableDimension="sagemaker:variant:DesiredInstanceCount",
    PolicyType="TargetTrackingScaling",
    TargetTrackingScalingPolicyConfiguration={
        "TargetValue": 750.0,        # Target: 750 invocations/minute per instance
        "PredefinedMetricSpecification": {
            "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance"
        },
        "ScaleInCooldown": 300,      # 5 min before scaling down
        "ScaleOutCooldown": 60,      # 1 min before scaling up
    },
)
```

---

## EC2 Deployment: Full Control

### EC2 Setup for ML API

```bash
# 1. Launch EC2 instance (g4dn.xlarge = 1 GPU, ~4 vCPUs, 16GB RAM)
# 2. Connect via SSH

# 3. Install dependencies
sudo apt update
sudo apt install python3-pip nginx docker.io -y
pip install fastapi uvicorn torch transformers

# 4. Clone your code
git clone https://github.com/your-org/ml-api.git
cd ml-api

# 5. Run with Docker (recommended)
docker build -t ml-api .
docker run -d -p 8000:8000 --gpus all ml-api

# 6. Configure Nginx as reverse proxy
sudo nano /etc/nginx/sites-available/ml-api
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/ml-api
upstream ml_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    location / {
        proxy_pass http://ml_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;

        # Timeouts for model inference
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
    }

    # Health check endpoint (no auth needed)
    location /health {
        proxy_pass http://ml_backend/health;
        access_log off;
    }
}
```

### Dockerfile for EC2 Deployment

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

WORKDIR /app

# Install Python
RUN apt-get update && apt-get install -y python3.11 python3-pip
RUN pip3 install --no-cache-dir torch torchvision transformers fastapi uvicorn

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

---

## EC2 vs SageMaker Comparison

```
┌───────────────────┬──────────────────┬──────────────────┐
│ Aspect            │ SageMaker        │ EC2              │
├───────────────────┼──────────────────┼──────────────────┤
│ Setup time        │ Minutes          │ Hours            │
│ Auto-scaling      │ Built-in         │ Manual/ASG       │
│ Model monitoring  │ Built-in         │ Custom           │
│ Cost (idle)       │ Stops endpoints  │ Pay per hour     │
│ GPU access        │ ml.g4dn, ml.p3   │ g4dn, p3, p4     │
│ Customization     │ Limited          │ Full control     │
│ A/B testing       │ Built-in         │ Custom           │
│ CI/CD integration │ SDK + Pipeline   │ Custom           │
│ Multi-model       │ Multi-model      │ Custom           │
│ Best for          │ Standard ML      │ Custom infra     │
│ Cost/hr (GPU)     │ ~$0.75 (g4dn.xl) │ ~$0.52 (g4dn.xl)│
└───────────────────┴──────────────────┴──────────────────┘
```

---

## Cost Optimization

### SageMaker Cost Strategies

```
┌─────────────────────────────────────────────┐
│  Cost Optimization Checklist                  │
│                                               │
│  ✓ Use Serverless for infrequent traffic     │
│  ✓ Use Spot instances for training           │
│  ✓ Right-size instances (don't over-provision)│
│  ✓ Enable auto-scaling with min=0            │
│  ✓ Use multi-model endpoints for small models │
│  ✓ Use Inferentia chips for cost efficiency   │
│  ✓ Schedule endpoints: stop at night          │
│  ✓ Use S3 Intelligent-Tiering for artifacts  │
└─────────────────────────────────────────────┘
```

### Spot Training

```python
# Use spot instances for training (up to 90% cheaper)
estimator = PyTorch(
    entry_point="train.py",
    role=role,
    framework_version="2.0",
    py_version="py310",
    instance_count=4,
    instance_type="ml.g4dn.xlarge",
    use_spot_instances=True,        # Use spot instances
    max_run=3600,                    # Max training time (1 hour)
    max_wait=7200,                   # Max wait for spot (2 hours)
    checkpoint_s3_uri="s3://bucket/checkpoints/",  # Resume from checkpoint
)
```

---

## Exercises

### Exercise 1: Choose the Right AWS Service
Match each scenario to the best AWS deployment option:
a) Real-time fraud detection with 10,000 req/sec
b) Daily batch scoring of 1M user features
c) Prototyping a new model locally
d) On-demand image generation with variable traffic

**Solution:**
a) **SageMaker real-time endpoint** with auto-scaling + GPU instances (ml.inf1 or ml.g5)
b) **SageMaker Batch Transform** or AWS Glue + Lambda for batch processing
c) **SageMaker Studio** notebook or local Jupyter with SageMaker Python SDK
d) **SageMaker Serverless Inference** or **EC2 with auto-scaling** (variable traffic → serverless)

### Exercise 2: Estimate Monthly Cost
You deploy a model on ml.m5.xlarge ($0.245/hr) with 2 instances, running 24/7. What's the monthly cost?

**Solution:**
```
Hourly: 2 × $0.245 = $0.49/hr
Daily: $0.49 × 24 = $11.76/day
Monthly: $11.76 × 30 = $352.80/month

With SageMaker auto-scaling (min=0, avg 8 hrs/day active):
$0.49 × 8 × 30 = $117.60/month (66% savings!)
```

### Exercise 3: Design a Blue-Green Deployment
How would you deploy a new model version without downtime?

**Solution:**
```python
# 1. Deploy new version to a separate endpoint variant
predictor_v2 = model_v2.deploy(
    endpoint_name="sentiment-endpoint",
    initial_instance_count=2,
    variant_name="VariantB",  # New variant
    update_endpoint=False,     # Don't replace existing
)

# 2. Shift traffic gradually (canary deployment)
sm_client.update_endpoint_weights_and_capacities(
    EndpointName="sentiment-endpoint",
    DesiredWeightsAndCapacities=[
        {"VariantName": "VariantA", "DesiredWeight": 90},  # Old: 90%
        {"VariantName": "VariantB", "DesiredWeight": 10},  # New: 10%
    ]
)

# 3. Monitor metrics, gradually shift to 100% new
# 4. Delete old variant when confident
```

---

## Key Takeaways

1. **SageMaker** provides managed deployment with auto-scaling, monitoring, and multi-model support
2. **EC2** offers full infrastructure control for custom deployments with Docker + Nginx
3. **Auto-scaling** adjusts capacity based on traffic—critical for cost optimization
4. **Spot instances** reduce training costs by up to 90% with checkpoint support
5. **Blue-green deployment** enables zero-downtime model updates via traffic shifting
