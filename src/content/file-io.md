# File I/O in Python

## Why File I/O Matters in AI/ML

Every ML project follows this lifecycle: **collect data → store data → read data → process data → write results**. You cannot train a model without first loading data from files. You cannot deploy a model without saving and loading trained weights. You cannot share results without writing reports to disk.

Consider the scale: a single training run might read millions of images from disk, stream terabytes of log data, and write checkpoints every hour. The Titanic dataset you train on in tutorials? Someone collected it, cleaned it, and saved it as a CSV — using file I/O.

This lesson covers reading and writing text files, CSV, JSON, and binary formats, plus context managers and real-world patterns for ML data pipelines.

---

## 1. Text Files — The Foundation

### 1.1 Reading Files

```python
# The basic way (requires manual closing)
f = open("data/sample.txt", "r")
content = f.read()
f.close()  # NEVER forget this — resource leaks are silent killers

# The Pythonic way — context manager (auto-closes)
with open("data/sample.txt", "r") as f:
    content = f.read()
# File is automatically closed when exiting the block

# Reading strategies for different data sizes
with open("data/large_dataset.txt", "r") as f:
    # Read entire file at once (small files only)
    # content = f.read()

    # Read line by line (memory-efficient for large files)
    for line in f:
        line = line.strip()
        if line:  # skip empty lines
            process(line)

    # Read all lines as list (adds \n to each line)
    # lines = f.readlines()

    # Read first N lines
    # lines = [f.readline().strip() for _ in range(10)]
```

### 1.2 Writing Files

```python
# Write mode (overwrites existing content)
with open("data/output.txt", "w") as f:
    f.write("Epoch 1, Loss: 0.923, Accuracy: 0.45\n")
    f.write("Epoch 2, Loss: 0.712, Accuracy: 0.62\n")

# Append mode (adds to existing content)
with open("data/training_log.txt", "a") as f:
    f.write(f"[{datetime.now()}] Epoch 100 completed\n")

# Write multiple lines at once
lines = [f"Feature: {name}, Importance: {imp:.4f}\n"
         for name, imp in zip(features, importances)]
with open("data/feature_report.txt", "w") as f:
    f.writelines(lines)
```

### 1.3 File Modes Reference

| Mode | Description | File exists? | File missing? |
|------|-------------|-------------|---------------|
| `"r"` | Read | Reads from start | **Error** |
| `"w"` | Write | **Truncates** (deletes content) | Creates new |
| `"a"` | Append | Writes at end | Creates new |
| `"r+"` | Read + Write | Reads and writes | **Error** |
| `"w+"` | Write + Read | Truncates, then read/write | Creates new |
| `"x"` | Exclusive create | **Error** | Creates new |
| `"rb"` / `"wb"` | Binary read/write | Same as above | Same as above |

---

## 2. CSV Files — Tabular Data

CSV (Comma-Separated Values) is the most common data format in ML. The UCI Machine Learning Repository hosts hundreds of CSV datasets. Kaggle competitions overwhelmingly use CSV for submission formats.

### 2.1 The csv Module (Standard Library)

```python
import csv

# READING CSV
with open("data/iris.csv", "r", newline="") as f:
    reader = csv.reader(f)
    headers = next(reader)  # Skip header row
    data = []
    for row in reader:
        # Convert numeric columns
        features = [float(x) for x in row[:4]]
        label = row[4]
        data.append({"features": features, "species": label})

print(f"Loaded {len(data)} samples")
print(f"First sample: {data[0]}")

# READING INTO DICTIONARIES (column access by name)
with open("data/iris.csv", "r", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # row is an OrderedDict: {'sepal_length': '5.1', ...}
        sepal_length = float(row["sepal_length"])
        species = row["species"]

# WRITING CSV
results = [
    {"model": "LogisticRegression", "accuracy": 0.96, "f1_score": 0.95},
    {"model": "RandomForest", "accuracy": 0.97, "f1_score": 0.97},
    {"model": "SVM", "accuracy": 0.98, "f1_score": 0.98},
]

with open("data/model_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "f1_score"])
    writer.writeheader()
    writer.writerows(results)
```

### 2.2 pandas (The ML Standard — preview)

```python
import pandas as pd

# One line replaces 15 lines of csv module code
df = pd.read_csv("data/iris.csv")
print(df.head())
print(df.describe())
print(df["species"].value_counts())

# Save results
df.to_csv("data/processed_iris.csv", index=False)
```

---

## 3. JSON Files — Structured Configuration and APIs

JSON (JavaScript Object Notation) maps directly to Python dictionaries and lists. It is the standard format for:
- ML model configurations (hyperparameters)
- Web API responses (REST APIs serving predictions)
- Training logs and experiment tracking
- NoSQL databases (MongoDB stores JSON-like documents)

### 3.1 Reading and Writing JSON

```python
import json

# ---- MODEL CONFIGURATION ----
config = {
    "experiment": "sentiment_analysis_v2",
    "model": {
        "type": "transformer",
        "pretrained": "bert-base-uncased",
        "fine_tune_layers": 4
    },
    "training": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 2e-5,
        "optimizer": "AdamW",
        "scheduler": {
            "type": "linear",
            "warmup_steps": 500
        }
    },
    "data": {
        "train_path": "data/train.jsonl",
        "val_path": "data/val.jsonl",
        "max_length": 128
    }
}

# Write JSON (pretty-printed for readability)
with open("config/experiment_config.json", "w") as f:
    json.dump(config, f, indent=2)

# Write JSON compact (for production — smaller files)
with open("config/experiment_compact.json", "w") as f:
    json.dump(config, f, separators=(",", ":"))

# Read JSON
with open("config/experiment_config.json", "r") as f:
    loaded_config = json.load(f)

print(loaded_config["training"]["learning_rate"])  # 2e-05
```

### 3.2 JSON Lines (JSONL) — Large Datasets

Each line is a separate JSON object. Standard format for NLP datasets (HuggingFace datasets often come in JSONL).

```python
# WRITING JSONL — streaming training data
training_samples = [
    {"text": "This movie was fantastic!", "label": 1, "source": "imdb"},
    {"text": "Terrible waste of time.", "label": 0, "source": "imdb"},
    {"text": "I loved every minute of it.", "label": 1, "source": "rotten_tomatoes"},
]

with open("data/train.jsonl", "w") as f:
    for sample in training_samples:
        f.write(json.dumps(sample) + "\n")

# READING JSONL — memory-efficient for large datasets
def stream_jsonl(filepath: str):
    """Generator that yields one JSON object per line."""
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# Process without loading everything into memory
positive_count = 0
total = 0
for sample in stream_jsonl("data/train.jsonl"):
    total += 1
    if sample["label"] == 1:
        positive_count += 1

print(f"Positive rate: {positive_count / total:.1%}")
```

### 3.3 JSON Serialization of Custom Objects

ML objects (models, numpy arrays) are not natively JSON-serializable. You need custom encoders:

```python
import numpy as np
from datetime import datetime

class MLEncoder(json.JSONEncoder):
    """Custom JSON encoder for ML objects."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

# Saving a training report with mixed types
report = {
    "timestamp": datetime.now(),
    "model": "RandomForest",
    "accuracy": float(np.float64(0.967)),
    "feature_importances": np.array([0.35, 0.28, 0.22, 0.15]),
    "unique_labels": {"cat", "dog", "bird"},
    "best_params": {"n_estimators": 200, "max_depth": 10}
}

with open("reports/training_report.json", "w") as f:
    json.dump(report, f, cls=MLEncoder, indent=2)
```

---

## 4. Context Managers — The `with` Statement

Context managers guarantee resource cleanup, even when exceptions occur. This is critical for ML pipelines where a crashed training job should not leave files locked or database connections open.

### 4.1 How Context Managers Work

```
    with open("data.txt", "r") as f:
    |   content = f.read()          # __enter__() called
    |   # ... operations ...        # your code runs here
    # end of block                  # __exit__() called automatically
                                    # file is closed, even if an error occurred
```

### 4.2 Built-in Context Managers

```python
# File I/O
with open("data.txt", "r") as f:
    ...

# Threading lock (prevents race conditions in parallel data loading)
from threading import Lock
lock = Lock()
with lock:
    shared_resource.update()

# tempfile — automatic cleanup
import tempfile
with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=True) as tmp:
    tmp.write("col1,col2\n1,2\n3,4\n")
    tmp.flush()
    # File exists at tmp.name while inside the block
    df = pd.read_csv(tmp.name)
# File is automatically deleted after the block
```

### 4.3 Building Custom Context Managers

```python
class ExperimentLogger:
    """Context manager for ML experiment logging."""

    def __init__(self, experiment_name: str, log_dir: str = "logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.log_file = None
        self.start_time = None
        self.metrics = []

    def __enter__(self):
        """Set up resources."""
        import os
        os.makedirs(self.log_dir, exist_ok=True)
        filepath = os.path.join(self.log_dir, f"{self.experiment_name}.log")
        self.log_file = open(filepath, "a")
        self.start_time = datetime.now()
        self._log(f"=== Experiment '{self.experiment_name}' started ===")
        return self  # 'as' variable gets this object

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources (always called, even on exceptions)."""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        if exc_type is None:
            self._log(f"=== Experiment completed in {elapsed:.2f}s ===")
        else:
            self._log(f"=== Experiment FAILED: {exc_val} ===")
        self.log_file.close()
        return False  # Don't suppress exceptions

    def _log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_file.write(f"[{timestamp}] {message}\n")
        self.log_file.flush()

    def log_metric(self, epoch: int, **metrics):
        metric_str = ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        self._log(f"Epoch {epoch:4d}: {metric_str}")
        self.metrics.append({"epoch": epoch, **metrics})


# Usage — resources are always cleaned up
with ExperimentLogger("bert_finetuning") as logger:
    for epoch in range(1, 6):
        logger.log_metric(epoch, loss=1.0 / epoch, accuracy=min(0.95, epoch * 0.2))
        if epoch == 3:
            # Even if this raises an error, the log file is closed properly
            pass
    print("Training completed!")
```

### 4.4 Context Manager with `contextlib`

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(name: str):
    """Measure execution time of a code block."""
    start = time.perf_counter()
    yield  # Code inside 'with' block runs here
    elapsed = time.perf_counter() - start
    print(f"[Timer] {name}: {elapsed:.4f}s")

# Usage
with timer("Loading dataset"):
    df = pd.read_csv("data/large_dataset.csv")

with timer("Training model"):
    model.fit(X_train, y_train)

with timer("Generating predictions"):
    predictions = model.predict(X_test)
```

---

## 5. Working with Directory Structures

ML projects have predictable directory layouts. Managing these programmatically is essential.

```
my_ml_project/
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   └── test.csv
│   ├── processed/
│   │   ├── train_clean.csv
│   │   └── test_clean.csv
│   └── external/
├── models/
│   ├── model_v1.pkl
│   └── model_v2.pkl
├── notebooks/
├── src/
├── logs/
└── reports/
```

```python
import os
from pathlib import Path

# OLD WAY (os module)
os.makedirs("data/processed", exist_ok=True)
for filename in os.listdir("data/raw"):
    if filename.endswith(".csv"):
        print(f"Found: {filename}")

# MODERN WAY (pathlib — object-oriented, chainable)
data_dir = Path("data")
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Find all CSV files recursively
csv_files = list(data_dir.rglob("*.csv"))
print(f"Found {len(csv_files)} CSV files")

# Copy and transform files
for csv_file in raw_dir.glob("*.csv"):
    output_path = processed_dir / f"clean_{csv_file.name}"
    print(f"Processing {csv_file.name} -> {output_path.name}")

# Check existence
if (processed_dir / "clean_train.csv").exists():
    print("Processed data found")

# Path operations
print(Path("data/raw/train.csv").stem)    # "train"
print(Path("data/raw/train.csv").suffix)  # ".csv"
print(Path("data/raw/train.csv").parent)  # "data/raw"
```

---

## 6. Pickle — Serializing Python Objects

Pickle serializes Python objects to binary. It is how scikit-learn saves models and how PyTorch saves state dictionaries.

```python
import pickle

# SAVE a trained model
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

model = LogisticRegression(max_iter=200)
iris = load_iris()
model.fit(iris.data, iris.target)

# Save
with open("models/iris_classifier.pkl", "wb") as f:
    pickle.dump(model, f)

# LOAD
with open("models/iris_classifier.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# Use the loaded model
predictions = loaded_model.predict(iris.data[:5])
print(predictions)  # Works identically to the original
```

### Pickle with Multiple Objects

```python
# Save an entire pipeline state
pipeline_state = {
    "model": model,
    "scaler": StandardScaler(),
    "feature_names": iris.feature_names,
    "target_names": iris.target_names.tolist(),
    "training_accuracy": model.score(iris.data, iris.target)
}

with open("models/pipeline_state.pkl", "wb") as f:
    pickle.dump(pipeline_state, f)

# Load
with open("models/pipeline_state.pkl", "rb") as f:
    state = pickle.load(f)

print(f"Model type: {type(state['model'])}")
print(f"Features: {state['feature_names']}")
```

> **Security Warning**: Never unpickle data from untrusted sources. Pickle can execute arbitrary code during deserialization. For safer alternatives, consider `joblib` (optimized for NumPy arrays) or `safetensors` (for PyTorch).

```python
# joblib — preferred for scikit-learn models (faster for NumPy)
import joblib
joblib.dump(model, "models/model.joblib")
model = joblib.load("models/model.joblib")
```

---

## 7. Real-World ML Pipeline: End-to-End File I/O

```python
import json
import csv
from pathlib import Path
from datetime import datetime

class DataPipeline:
    """Complete data pipeline with file I/O at every stage."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.data_dir = Path(self.config["data_dir"])

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)

    def load_raw_data(self) -> list:
        """Load raw CSV data into memory."""
        filepath = self.data_dir / "raw" / self.config["raw_filename"]
        data = []
        with open(filepath, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                processed_row = {}
                for key, value in row.items():
                    try:
                        processed_row[key] = float(value)
                    except ValueError:
                        processed_row[key] = value
                data.append(processed_row)
        print(f"Loaded {len(data)} rows from {filepath}")
        return data

    def clean_data(self, data: list) -> list:
        """Remove rows with missing values."""
        cleaned = [row for row in data
                   if all(v != "" and v is not None for v in row.values())]
        removed = len(data) - len(cleaned)
        print(f"Removed {removed} rows with missing values")
        return cleaned

    def save_processed(self, data: list) -> Path:
        """Save cleaned data as CSV."""
        output_dir = self.data_dir / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"cleaned_{timestamp}.csv"

        if data:
            with open(output_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=data[0].keys())
                writer.writeheader()
                writer.writerows(data)

        print(f"Saved {len(data)} rows to {output_path}")
        return output_path

    def save_report(self, stats: dict):
        """Save experiment report as JSON."""
        report_dir = self.data_dir.parent / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "statistics": stats
        }

        output_path = report_dir / f"report_{datetime.now():%Y%m%d}.json"
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Report saved to {output_path}")


# Usage
pipeline = DataPipeline("config/pipeline_config.json")
raw_data = pipeline.load_raw_data()
clean_data = pipeline.clean_data(raw_data)
pipeline.save_processed(clean_data)
pipeline.save_report({"rows": len(clean_data), "columns": len(clean_data[0])})
```

---

## 8. Handling Large Files — Memory-Efficient Patterns

```python
def process_large_csv(filepath: str, chunk_size: int = 10000, output_path: str = None):
    """
    Process a large CSV file in chunks without loading it entirely into memory.
    This pattern is essential when datasets exceed available RAM (common in real ML).
    """
    chunk_stats = []

    with open(filepath, "r", newline="") as f:
        reader = csv.DictReader(f)
        chunk = []
        chunk_num = 0

        for i, row in enumerate(reader):
            chunk.append(row)

            if len(chunk) >= chunk_size:
                # Process this chunk
                numeric_values = []
                for r in chunk:
                    for v in r.values():
                        try:
                            numeric_values.append(float(v))
                        except ValueError:
                            pass

                stats = {
                    "chunk": chunk_num,
                    "rows": len(chunk),
                    "numeric_mean": sum(numeric_values) / len(numeric_values) if numeric_values else 0
                }
                chunk_stats.append(stats)

                # Optionally write processed chunk
                if output_path and chunk_num == 0:
                    with open(output_path, "w", newline="") as out:
                        writer = csv.DictWriter(out, fieldnames=chunk[0].keys())
                        writer.writeheader()
                        writer.writerows(chunk)
                elif output_path:
                    with open(output_path, "a", newline="") as out:
                        writer = csv.DictWriter(out, fieldnames=chunk[0].keys())
                        writer.writerows(chunk)

                print(f"  Processed chunk {chunk_num}: {len(chunk)} rows")
                chunk = []
                chunk_num += 1

        # Don't forget the last partial chunk
        if chunk:
            print(f"  Processed final chunk {chunk_num}: {len(chunk)} rows")

    return chunk_stats

# Alternative: pandas chunked reading (covered in Pandas lesson)
# for chunk in pd.read_csv("huge_file.csv", chunksize=10000):
#     process(chunk)
```

---

## 9. Exercises

### Exercise 1: Build a Training Logger

Create a logger that writes training metrics to both a CSV file and a JSON file simultaneously.

```python
class DualLogger:
    def __init__(self, csv_path: str, json_path: str):
        self.csv_path = csv_path
        self.json_path = json_path
        self.history = []
        self._init_csv()

    def _init_csv(self):
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "train_acc", "val_acc"])

    def log(self, epoch, train_loss, val_loss, train_acc, val_acc):
        entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4)
        }
        self.history.append(entry)

        # Append to CSV
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(list(entry.values()))

    def save_json(self):
        with open(self.json_path, "w") as f:
            json.dump(self.history, f, indent=2)

# Usage
logger = DualLogger("logs/training.csv", "logs/training.json")
for epoch in range(1, 11):
    logger.log(epoch, 1.0/epoch, 1.2/epoch, min(0.99, epoch*0.1), min(0.95, epoch*0.09))
logger.save_json()
```

### Exercise 2: JSON Config Validator

Write a function that loads a JSON config file and validates that all required keys exist with correct types.

### Exercise 3: File Format Converter

Build a tool that converts between CSV, JSON, and JSONL formats for the same dataset.

---

## Connections to Other ML Topics

| File I/O Concept | ML Application |
|-----------------|----------------|
| CSV reading | Loading Kaggle datasets, UCI repository data |
| JSON config | Experiment tracking (MLflow, Weights & Biases) |
| JSONL streaming | Processing large NLP datasets (Common Crawl, Wikipedia) |
| Pickle/joblib | Saving trained models for deployment |
| Context managers | Resource management in data pipelines |
| pathlib | Organizing project directories |
| Chunked reading | Training on datasets larger than RAM |

---

## Key Takeaways

1. **Always use context managers** (`with open(...) as f`) — never raw `open()` without guaranteed closing.
2. **CSV** is the universal tabular format; `csv.DictReader` maps columns by name.
3. **JSON** is the standard for configurations and API data; `json.dump(load)` with `indent=2` for readability.
4. **JSONL** handles large datasets line-by-line without loading everything into memory.
5. **Pickle/joblib** serializes Python objects — essential for model persistence.
6. **pathlib** is the modern, object-oriented replacement for `os.path`.
7. **Chunked processing** is mandatory when datasets exceed available RAM.
