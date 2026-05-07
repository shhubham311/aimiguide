# Programming in Python & Basic Data Structures

## Why Python Matters in AI/ML

Python is the lingua franca of artificial intelligence and machine learning. When researchers at Google Brain published the original Transformer paper ("Attention Is All You Need"), the reference implementation was written in Python using TensorFlow. When OpenAI built GPT, the training pipeline was Python. When you train a neural network with PyTorch, write a data pipeline with pandas, or deploy a model with FastAPI — it is all Python.

The reasons are not accidental:

- **Readability**: Python's syntax mirrors pseudocode, letting you express algorithms close to their mathematical definition. A gradient descent step in Python reads almost like the textbook formula.
- **Ecosystem**: NumPy, SciPy, pandas, scikit-learn, PyTorch, TensorFlow, Jupyter — the entire ML toolchain is Python-first.
- **Rapid prototyping**: An idea that would take weeks in C++ can be tested in Python in hours.
- **Community**: Kaggle competitions, research papers, open-source libraries — the community converges on Python.

This lesson covers the foundational building blocks: variables, control flow, and the four core data structures (lists, tuples, sets, dictionaries) that you will use in every ML pipeline.

---

## 1. Variables and Types

Python is dynamically typed. You do not declare types; the interpreter infers them at runtime.

```python
# Variables — the atoms of every program
learning_rate = 0.001          # float
epochs = 100                   # int
model_name = "resnet50"        # str
use_dropout = True             # bool
hidden_sizes = [128, 64, 32]   # list

# Type checking (runtime, not compile-time)
print(type(learning_rate))     # <class 'float'>
print(isinstance(epochs, int)) # True

# Multiple assignment — handy for unpacking
train_acc, val_acc, test_acc = 0.95, 0.91, 0.89
```

### Dynamic typing vs. Static typing in ML

Most ML code uses dynamic typing for speed of iteration. However, for production pipelines, Python 3.6+ supports type hints that help catch bugs early and enable static analysis with tools like `mypy`:

```python
from typing import List, Dict, Optional

def train_model(
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 100,
    learning_rate: float = 0.001,
    verbose: bool = True
) -> Dict[str, List[float]]:
    """Train a model and return metrics history."""
    ...
```

---

## 2. Control Flow

Control flow dictates how your program decides and repeats — the backbone of every algorithm.

### 2.1 Conditional Logic

```python
# Activation function selection — a common ML pattern
def activate(x: float, activation: str = "relu") -> float:
    if activation == "relu":
        return max(0.0, x)
    elif activation == "sigmoid":
        return 1.0 / (1.0 + (-x) ** 2)  # simplified
    elif activation == "tanh":
        import math
        return math.tanh(x)
    else:
        raise ValueError(f"Unknown activation: {activation}")

print(activate(-2.0, "relu"))    # 0.0
print(activate(0.5, "sigmoid"))  # 0.8
```

### 2.2 Loops

```python
# Computing loss over batches
batch_losses = [0.45, 0.38, 0.31, 0.27, 0.25]

# for loop — iterate directly
total_loss = 0.0
for loss in batch_losses:
    total_loss += loss
avg_loss = total_loss / len(batch_losses)

# enumerate — when you need the index
for i, loss in enumerate(batch_losses):
    if i > 0 and batch_losses[i-1] - loss < 0.01:
        print(f"Convergence slowing at batch {i}")

# while loop — gradient descent from scratch
x = 10.0          # initial weight
lr = 0.1          # learning rate
target = 3.0      # target value

for step in range(50):
    gradient = 2 * (x - target)  # derivative of (x - target)^2
    x -= lr * gradient
    if abs(x - target) < 1e-6:
        print(f"Converged in {step} steps: x = {x:.6f}")
        break
```

### Control Flow Diagram

```
                    START
                      |
                 +----v----+
                 | Condition|
                 |  met?    |
                 +----+----+
               Yes/    \No
                /       \
         +----v----+   +--v----------+
         | Action A |   | Action B   |
         +----+----+   +------+------+
              |               |
              +-------+-------+
                      |
                   +--v--+
                   |Next |
                   +-----+
```

---

## 3. Lists

Lists are ordered, mutable sequences — the workhorse of Python data handling.

```python
# Creating lists
features = ["age", "income", "education"]
predictions = [0.2, 0.8, 0.1, 0.95, 0.3]

# Indexing and slicing (0-based, end-exclusive)
print(features[0])       # "age"
print(features[-1])      # "education"
print(features[1:])      # ["income", "education"]
print(predictions[::2])  # [0.2, 0.1, 0.3]  (every 2nd)

# Mutation
features.append("credit_score")
features.insert(1, "gender")
removed = features.pop()  # removes and returns last

# List methods you will use constantly in ML
data = [3, 1, 4, 1, 5, 9, 2, 6]
print(len(data))           # 8
print(sum(data))           # 31
print(min(data), max(data))# 1, 9
data.sort()                # in-place sort: [1, 1, 2, 3, 4, 5, 6, 9]
print(data.count(1))       # 2
```

### Nested Lists — Matrices

Before NumPy, people used nested lists for matrix operations:

```python
# A 2x3 matrix as a list of lists
matrix = [
    [1, 2, 3],
    [4, 5, 6]
]

# Access row 0, column 1
print(matrix[0][1])  # 2

# Naive matrix transpose (illustrates why NumPy is better)
transpose = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
# Result: [[1, 4], [2, 5], [3, 6]]
```

---

## 4. Tuples

Tuples are ordered, **immutable** sequences. Once created, they cannot be changed.

```python
# Model configuration — use tuples when data should not change
IMG_SHAPE = (224, 224, 3)   # height, width, channels
LABELS = ("cat", "dog", "bird")
RGB_RANGE = (0, 255)

# Tuple unpacking — extremely common in ML
weights = np.array([0.5, -0.3, 0.8])
bias = 0.1
# In training loops you often unpack tuples:
for epoch, (train_loss, val_loss) in enumerate(history):
    print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

# Single-element tuple (note the comma)
single = (42,)    # tuple
not_tuple = (42)  # just an int
```

### When to use tuples vs lists

| Aspect | List | Tuple |
|--------|------|-------|
| Mutability | Mutable | Immutable |
| Performance | Slightly slower | Slightly faster |
| Use case | Dynamic collections | Fixed structure, dict keys |
| ML example | `predictions = []` | `input_shape = (28, 28, 1)` |

---

## 5. Sets

Sets are unordered collections of **unique** elements. They implement mathematical set operations.

```python
# Removing duplicate features
raw_features = ["age", "income", "age", "education", "income"]
unique_features = set(raw_features)
# {'age', 'income', 'education'}

# Set operations — finding common/different items
model_a_features = {"age", "income", "credit_score", "education"}
model_b_features = {"income", "education", "employment_status"}

print(model_a_features & model_b_features)  # intersection
# {'income', 'education'}

print(model_a_features - model_b_features)  # difference
# {'age', 'credit_score'}

print(model_a_features | model_b_features)  # union
# {'age', 'income', 'credit_score', 'education', 'employment_status'}
```

### Real ML use case: Feature overlap analysis

```python
def feature_overlap_check(datasets: dict) -> None:
    """Analyze feature overlap across multiple datasets."""
    feature_sets = {name: set(features) for name, features in datasets.items()}
    names = list(feature_sets.keys())

    print("=" * 50)
    print("FEATURE OVERLAP MATRIX")
    print("=" * 50)
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if i < j:
                common = feature_sets[n1] & feature_sets[n2]
                only_n1 = feature_sets[n1] - feature_sets[n2]
                only_n2 = feature_sets[n2] - feature_sets[n1]
                print(f"\n{n1} vs {n2}:")
                print(f"  Common: {len(common)}, Only in {n1}: {len(only_n1)}, Only in {n2}: {len(only_n2)}")

# Usage
feature_overlap_check({
    "census": ["age", "income", "education", "occupation", "hours_per_week"],
    "credit": ["age", "income", "credit_score", "loan_amount", "default_history"],
    "health": ["age", "bmi", "blood_pressure", "glucose", "education"]
})
```

---

## 6. Dictionaries

Dictionaries are key-value mappings — the single most important data structure for ML configuration and structured data.

```python
# Model hyperparameters — the standard pattern
config = {
    "model": "random_forest",
    "n_estimators": 100,
    "max_depth": 10,
    "min_samples_split": 5,
    "random_state": 42,
    "features": ["age", "income", "education"]
}

# Access patterns
print(config["model"])                    # direct access (KeyError if missing)
print(config.get("learning_rate", 0.01))  # safe access with default

# Iteration
for key in config:
    print(f"{key}: {config[key]}")

for key, value in config.items():
    print(f"{key} = {value}")

# Mutation
config["n_estimators"] = 200
config.update({"criterion": "gini", "bootstrap": True})

# Deletion
del config["random_state"]

# Nested dictionaries — realistic ML config
pipeline_config = {
    "data": {
        "source": "s3://bucket/dataset.csv",
        "test_size": 0.2,
        "random_state": 42
    },
    "preprocessing": {
        "numeric": {"strategy": "standard"},
        "categorical": {"strategy": "onehot"}
    },
    "model": {
        "type": "xgboost",
        "params": {
            "n_estimators": 500,
            "max_depth": 6,
            "learning_rate": 0.05
        }
    }
}
```

### Dictionary comprehensions

```python
# Count word frequencies in a document (NLP preprocessing)
text = "machine learning is the study of computer algorithms that improve automatically through experience"
word_counts = {word: text.split().count(word) for word in set(text.split())}
# {'machine': 1, 'learning': 1, 'is': 1, 'the': 1, ...}

# Feature importance ranking
importances = {"age": 0.35, "income": 0.28, "education": 0.15, "credit": 0.22}
ranked = {k: v for k, v in sorted(importances.items(), key=lambda x: x[1], reverse=True)}
# {'age': 0.35, 'income': 0.28, 'credit': 0.22, 'education': 0.15}
```

---

## 7. List Comprehensions

List comprehensions are Pythonic, concise, and often faster than equivalent for-loops. They are essential for data preprocessing.

```python
# Basic syntax: [expression for item in iterable if condition]

# Filter predictions above threshold
probabilities = [0.1, 0.8, 0.3, 0.95, 0.2, 0.7]
high_conf = [p for p in probabilities if p > 0.5]
# [0.8, 0.95, 0.7]

# Transform and filter in one step
scores = [85, 92, 45, 78, 99, 33, 67]
passed_a = [s * 1.05 for s in scores if s >= 50]  # 5% bonus for passing
# [89.25, 96.6, 81.9, 103.95, 70.35]

# Nested list comprehension — flatten a confusion matrix
confusion = [[50, 5], [3, 42]]
flat = [cell for row in confusion for cell in row]
# [50, 5, 3, 42]

# Dict comprehension from two lists
feature_names = ["petal_length", "petal_width", "sepal_length", "sepal_width"]
feature_importances = [0.42, 0.31, 0.18, 0.09]
importance_map = {name: imp for name, imp in zip(feature_names, feature_importances)}
# {'petal_length': 0.42, 'petal_width': 0.31, ...}

# Set comprehension — unique labels
labels = ["cat", "dog", "cat", "bird", "dog", "cat"]
unique_labels = {label for label in labels}
# {'cat', 'dog', 'bird'}
```

### Performance comparison

```
Task: Square all even numbers from 0 to 999,999

List comprehension:  ~45 ms
For loop with append: ~85 ms
Map + lambda:         ~65 ms
NumPy vectorized:     ~3 ms   (covered in the NumPy lesson)
```

---

## 8. The `zip()` Function and Other Essentials

```python
# zip — pair elements from multiple sequences
names = ["Alice", "Bob", "Charlie"]
scores = [95, 87, 92]
rankings = [1, 3, 2]

for name, score, rank in zip(names, scores, rankings):
    print(f"#{rank} {name}: {score}")

# enumerate — get index + value
features = ["age", "income", "education"]
for i, feat in enumerate(features):
    print(f"Feature {i}: {feat}")

# map and filter — functional alternatives
data = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, data))       # [1, 4, 9, 16, 25]
evens = list(filter(lambda x: x % 2 == 0, data)) # [2, 4]

# any and all — useful for validation checks
predictions = [0.9, 0.8, 0.1, 0.95]
print(any(p > 0.9 for p in predictions))  # True
print(all(p > 0.5 for p in predictions))  # False
```

---

## 9. Putting It All Together: A Mini ML Pipeline

```python
def simple_knn_classifier(
    train_data: list,
    train_labels: list,
    test_point: list,
    k: int = 3
) -> str:
    """
    A from-scratch k-NN classifier using only basic Python.
    Demonstrates: lists, dictionaries, comprehensions, zip, sorted().
    """
    # Step 1: Compute distances
    distances = []
    for i, train_pt in enumerate(train_data):
        dist = sum((a - b) ** 2 for a, b in zip(train_pt, test_point)) ** 0.5
        distances.append((dist, train_labels[i]))

    # Step 2: Sort by distance
    distances.sort(key=lambda x: x[0])

    # Step 3: Get k nearest labels
    nearest_labels = [label for _, label in distances[:k]]

    # Step 4: Majority vote using dictionary counting
    vote_counts = {}
    for label in nearest_labels:
        vote_counts[label] = vote_counts.get(label, 0) + 1

    # Step 5: Return the label with most votes
    return max(vote_counts, key=vote_counts.get)

# --- Usage with Iris-like data ---
# Features: [petal_length, petal_width]
training_data = [
    [1.4, 0.2], [1.3, 0.2], [1.5, 0.3],  # Setosa
    [4.7, 1.4], [4.5, 1.5], [4.9, 1.5],  # Versicolor
    [6.0, 2.5], [5.9, 2.1], [6.1, 2.3]   # Virginica
]
training_labels = ["setosa"] * 3 + ["versicolor"] * 3 + ["virginica"] * 3

test_point = [4.8, 1.6]
prediction = simple_knn_classifier(training_data, training_labels, test_point, k=3)
print(f"Predicted: {prediction}")  # Predicted: versicolor
```

---

## 10. Exercises

### Exercise 1: Confusion Matrix from Scratch

Given two lists `y_true` and `y_pred`, build a confusion matrix as a nested dictionary.

```python
y_true = ["cat", "dog", "cat", "bird", "dog", "cat", "bird", "cat"]
y_pred = ["cat", "dog", "bird", "bird", "cat", "cat", "bird", "dog"]

# Expected output:
# {
#   ("cat", "cat"): 2, ("cat", "dog"): 1, ("cat", "bird"): 0,
#   ("dog", "cat"): 1, ("dog", "dog"): 1, ("dog", "bird"): 0,
#   ("bird", "cat"): 0, ("bird", "dog"): 0, ("bird", "bird"): 2
# }
```

**Solution:**

```python
labels = list(set(y_true) | set(y_pred))
confusion = {(true, pred): 0 for true in labels for pred in labels}
for t, p in zip(y_true, y_pred):
    confusion[(t, p)] += 1

# Print formatted
print(f"{'':>8}", end="")
for l in labels:
    print(f"{l:>8}", end="")
print()
for t in labels:
    print(f"{t:>8}", end="")
    for p in labels:
        print(f"{confusion[(t,p)]:>8}", end="")
    print()
```

### Exercise 2: Feature Standardization

Implement z-score standardization using only lists and comprehensions (no NumPy).

```python
data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# z = (x - mean) / std
mean = sum(data) / len(data)
variance = sum((x - mean) ** 2 for x in data) / len(data)
std = variance ** 0.5
standardized = [(x - mean) / std for x in data]

print(f"Mean: {mean}, Std: {std}")
print(f"Standardized: {[round(s, 4) for s in standardized]}")
```

### Exercise 3: One-Hot Encoding

Convert a list of categorical labels into one-hot encoded vectors using lists.

```python
labels = ["cat", "dog", "bird", "cat", "dog"]
classes = sorted(set(labels))  # ['bird', 'cat', 'dog']

one_hot = []
for label in labels:
    vec = [1 if label == cls else 0 for cls in classes]
    one_hot.append(vec)

# Result:
# [[0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
print(one_hot)
```

---

## Connections to Other ML Topics

| This Lesson | Connects To |
|------------|-------------|
| Lists, loops | NumPy arrays (vectorized operations replace loops) |
| Dictionaries | Pandas DataFrames (dict-like column access) |
| List comprehensions | Feature engineering pipelines |
| Tuples | Model shapes, configuration immutability |
| Sets | Feature deduplication, categorical analysis |
| Control flow | Training loops, early stopping logic |
| `zip()` | Batching data, pairing inputs with labels |

---

## Key Takeaways

1. **Master the four core structures**: Lists (mutable sequences), Tuples (immutable), Sets (unique elements), Dictionaries (key-value maps).
2. **List comprehensions** are Pythonic, fast, and readable — use them for data transforms.
3. **Dictionaries** are the backbone of ML configuration and structured data.
4. **Control flow** (if/elif/else, for, while, break, continue) implements every algorithm's logic.
5. **Everything you build here** scales directly into NumPy, Pandas, and beyond.
