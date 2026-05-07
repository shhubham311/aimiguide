# OOPS in Python & Examples

## Why Object-Oriented Programming Matters in AI/ML

Every major ML framework is built on OOP. When you write `model = LogisticRegression()`, you are instantiating an object. When you call `model.fit(X, y)`, you are invoking a method. When you access `model.coef_`, you are reading an attribute. Scikit-learn's entire API design is an OOP masterclass:

```
model = SomeEstimator(params)     # Constructor
model.fit(X_train, y_train)        # Training method
predictions = model.predict(X)     # Inference method
score = model.score(X, y)          # Evaluation method
```

Understanding OOP is not optional for ML engineers — it is how you read, use, extend, and build ML systems. Custom loss functions, data loaders, model architectures in PyTorch, pipeline components in scikit-learn — all require OOP.

---

## 1. Classes and Objects

A **class** is a blueprint. An **object** (instance) is a concrete realization of that blueprint.

```python
class DataPoint:
    """Represents a single data point in a dataset."""

    def __init__(self, features: list, label: str):
        self.features = features      # instance attribute
        self.label = label            # instance attribute
        self.is_processed = False     # instance attribute with default

    def process(self):
        """Process this data point."""
        if not self.is_processed:
            # Normalize features to [0, 1]
            min_val = min(self.features)
            max_val = max(self.features)
            if max_val > min_val:
                self.features = [(f - min_val) / (max_val - min_val)
                                  for f in self.features]
            self.is_processed = True

    def __repr__(self):
        return f"DataPoint(label={self.label}, features={self.features})"

# Creating objects
dp1 = DataPoint([10, 20, 30], "class_A")
dp2 = DataPoint([5, 15, 25], "class_B")

print(dp1)                    # DataPoint(label=class_A, features=[10, 20, 30])
print(dp1.is_processed)       # False
dp1.process()
print(dp1.is_processed)       # True
print(dp1.features)           # [0.0, 0.5, 1.0]
```

### Class vs Instance Attributes

```
+-------------------------------------------+
|              ClassName                     |
|-------------------------------------------|
|  CLASS ATTRIBUTES (shared by all)         |
|    class_var = "shared"                    |
|-------------------------------------------|
|  def __init__(self, ...):                  |
|    self.instance_var = "unique per obj"   |
+-------------------------------------------+
        |                    |
   +----v------+       +----v------+
   | Object A  |       | Object B  |
   | inst="a"  |       | inst="b"  |
   +-----------+       +-----------+
   Both share: class_var = "shared"
```

```python
class ModelRegistry:
    """Track all model instances created."""
    count = 0  # class attribute

    def __init__(self, name: str):
        ModelRegistry.count += 1
        self.id = ModelRegistry.count  # instance attribute
        self.name = name

    @classmethod
    def total_models(cls):
        return cls.count

m1 = ModelRegistry("logistic_regression")
m2 = ModelRegistry("random_forest")
m3 = ModelRegistry("neural_network")

print(m1.id, m2.id, m3.id)          # 1 2 3
print(ModelRegistry.total_models()) # 3
```

---

## 2. Encapsulation

Encapsulation bundles data and methods that operate on that data, and restricts direct access to internal state. Python uses **convention** (underscore prefixes) rather than strict access modifiers.

```python
class NeuralNetworkLayer:
    """A single layer in a neural network."""

    def __init__(self, n_inputs: int, n_outputs: int):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self._initialize_weights()  # "protected" — convention

    def _initialize_weights(self):
        """Internal method — should not be called directly."""
        import random
        # Xavier initialization
        limit = (6.0 / (self.n_inputs + self.n_outputs)) ** 0.5
        self._weights = [
            [random.uniform(-limit, limit) for _ in range(self.n_inputs)]
            for _ in range(self.n_outputs)
        ]
        self._bias = [0.0] * self.n_outputs

    def forward(self, inputs: list) -> list:
        """Public API — compute layer output."""
        outputs = []
        for i in range(self.n_outputs):
            z = sum(w * x for w, x in zip(self._weights[i], inputs)) + self._bias[i]
            outputs.append(max(0.0, z))  # ReLU activation
        return outputs

    def get_weights_summary(self) -> dict:
        """Controlled access to internal state."""
        return {
            "shape": (self.n_outputs, self.n_inputs),
            "mean_weight": sum(w for row in self._weights for w in row) /
                           (self.n_outputs * self.n_inputs),
            "bias": self._bias.copy()
        }

layer = NeuralNetworkLayer(3, 2)
output = layer.forward([1.0, 2.0, 3.0])
print(f"Output: {output}")
print(f"Summary: {layer.get_weights_summary()}")
# Direct access to _weights works but is discouraged:
# layer._weights  # You CAN do this in Python, but you SHOULDN'T
```

### Naming conventions

| Prefix | Meaning | Example | Enforced? |
|--------|---------|---------|-----------|
| (none) | Public | `forward()` | No |
| `_` | Protected (convention) | `_weights` | No — just a convention |
| `__` | Name-mangled | `__private_var` | Partially — becomes `_ClassName__private_var` |
| `__x__` | Magic/Dunder | `__init__`, `__repr__` | Used by Python internally |

---

## 3. Inheritance

Inheritance lets you create specialized classes from general ones, avoiding code duplication.

```python
class BaseClassifier:
    """Abstract base for all classifiers."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
        self._training_data = None

    def fit(self, X: list, y: list):
        """Must be overridden by subclasses."""
        raise NotImplementedError(f"{self.name} must implement fit()")

    def predict(self, X: list) -> list:
        """Must be overridden by subclasses."""
        raise NotImplementedError(f"{self.name} must implement predict()")

    def evaluate(self, X: list, y: list) -> dict:
        """Common evaluation — uses predict() from subclass."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before evaluation")
        predictions = self.predict(X)
        correct = sum(1 for p, t in zip(predictions, y) if p == t)
        accuracy = correct / len(y)
        return {"accuracy": accuracy, "total": len(y), "correct": correct}


class KNearestNeighbors(BaseClassifier):
    """k-NN classifier inheriting from BaseClassifier."""

    def __init__(self, k: int = 3):
        super().__init__("KNearestNeighbors")
        self.k = k
        self._train_X = None
        self._train_y = None

    def fit(self, X: list, y: list):
        self._train_X = X
        self._train_y = y
        self.is_fitted = True
        return self

    def predict(self, X: list) -> list:
        predictions = []
        for test_pt in X:
            distances = [
                (sum((a - b) ** 2 for a, b in zip(train_pt, test_pt)) ** 0.5, label)
                for train_pt, label in zip(self._train_X, self._train_y)
            ]
            distances.sort(key=lambda x: x[0])
            nearest = [label for _, label in distances[:self.k]]
            # Majority vote
            votes = {}
            for label in nearest:
                votes[label] = votes.get(label, 0) + 1
            predictions.append(max(votes, key=votes.get))
        return predictions


class DecisionStump(BaseClassifier):
    """Simple one-level decision tree for demonstration."""

    def __init__(self):
        super().__init__("DecisionStump")
        self.best_feature = None
        self.best_threshold = None
        self.best_left_label = None
        self.best_right_label = None

    def fit(self, X: list, y: list):
        best_accuracy = 0
        n_features = len(X[0])

        for feature_idx in range(n_features):
            values = sorted(set(row[feature_idx] for row in X))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2

                left_labels = [y[j] for j in range(len(X))
                               if X[j][feature_idx] <= threshold]
                right_labels = [y[j] for j in range(len(X))
                                if X[j][feature_idx] > threshold]

                if left_labels and right_labels:
                    from collections import Counter
                    left_pred = Counter(left_labels).most_common(1)[0][0]
                    right_pred = Counter(right_labels).most_common(1)[0][0]
                    preds = [left_pred if X[j][feature_idx] <= threshold
                             else right_pred for j in range(len(X))]
                    acc = sum(1 for p, t in zip(preds, y) if p == t) / len(y)

                    if acc > best_accuracy:
                        best_accuracy = acc
                        self.best_feature = feature_idx
                        self.best_threshold = threshold
                        self.best_left_label = left_pred
                        self.best_right_label = right_pred

        self.is_fitted = True
        return self

    def predict(self, X: list) -> list:
        return [
            self.best_left_label if row[self.best_feature] <= self.best_threshold
            else self.best_right_label
            for row in X
        ]


# --- Usage ---
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data.tolist()[:100]  # Only 2 classes for simplicity
y = [str(label) for label in iris.target[:100]]

# Split
split = 80
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

knn = KNearestNeighbors(k=5)
knn.fit(X_train, y_train)
print(f"k-NN: {knn.evaluate(X_test, y_test)}")

stump = DecisionStump()
stump.fit(X_train, y_train)
print(f"Stump: {stump.evaluate(X_test, y_test)}")
```

### Inheritance Hierarchy in scikit-learn

```
BaseEstimator
├── ClassifierMixin
│   ├── LogisticRegression
│   ├── RandomForestClassifier
│   └── SVC
├── RegressorMixin
│   ├── LinearRegression
│   └── RandomForestRegressor
└── TransformerMixin
    ├── StandardScaler
    └── PCA
```

Every scikit-learn estimator inherits from `BaseEstimator` which provides `get_params()` and `set_params()` automatically.

---

## 4. Polymorphism

Polymorphism means "many forms" — different objects respond to the same interface in their own way.

```python
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    """
    This function works with ANY classifier that implements
    fit(X, y) and predict(X) — that is polymorphism.
    """
    model.fit(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    print(f"{model.name}: accuracy = {results['accuracy']:.2%}")
    return results

# Same function, different behaviors
train_and_evaluate(KNearestNeighbors(k=3), X_train, y_train, X_test, y_test)
train_and_evaluate(DecisionStump(), X_train, y_train, X_test, y_test)

# Duck typing — "if it walks like a duck and quacks like a duck..."
# Python does not require formal inheritance for polymorphism.
# Any object with fit() and predict() methods works.
```

### Operator Overloading

Python lets you redefine how operators work with your objects via dunder methods:

```python
class Vector:
    """A mathematical vector supporting arithmetic operations."""

    def __init__(self, components: list):
        self.components = list(components)

    def __add__(self, other):
        """Vector addition: self + other"""
        if len(self) != len(other):
            raise ValueError("Vectors must have same dimension")
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __mul__(self, scalar):
        """Scalar multiplication: vector * scalar"""
        return Vector([c * scalar for c in self.components])

    def __rmul__(self, scalar):
        """Scalar multiplication: scalar * vector"""
        return self.__mul__(scalar)

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def __repr__(self):
        return f"Vector({self.components})"

    def __len__(self):
        return len(self.components)

    def dot(self, other) -> float:
        """Dot product."""
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self) -> float:
        return self.dot(self) ** 0.5

# This makes linear algebra operations feel natural
w = Vector([0.5, -0.3, 0.8])
b = Vector([0.1, 0.2, -0.1])
x = Vector([2.0, 3.0, 1.0])

print(w + b)           # Vector([0.6, -0.1, 0.7])
print(2 * x)           # Vector([4.0, 6.0, 2.0])
print(w.dot(x))        # 1.5
```

---

## 5. Abstraction

Abstraction hides complexity behind simple interfaces. Users interact with high-level operations without knowing the implementation details.

```python
from abc import ABC, abstractmethod

class DataPreprocessor(ABC):
    """Abstract base class for data preprocessing pipelines."""

    @abstractmethod
    def fit(self, data: list) -> 'DataPreprocessor':
        """Learn parameters from training data."""
        pass

    @abstractmethod
    def transform(self, data: list) -> list:
        """Apply learned transformation."""
        pass

    def fit_transform(self, data: list) -> list:
        """Convenience method: fit then transform."""
        return self.fit(data).transform(data)


class StandardScaler(DataPreprocessor):
    """Z-score standardization: z = (x - mean) / std."""

    def __init__(self):
        self._means = None
        self._stds = None

    def fit(self, data: list) -> 'StandardScaler':
        n_features = len(data[0])
        self._means = []
        self._stds = []
        for j in range(n_features):
            col = [row[j] for row in data]
            mean = sum(col) / len(col)
            var = sum((x - mean) ** 2 for x in col) / len(col)
            std = var ** 0.5 if var > 0 else 1.0
            self._means.append(mean)
            self._stds.append(std)
        return self

    def transform(self, data: list) -> list:
        if self._means is None:
            raise RuntimeError("Scaler not fitted")
        return [
            [(row[j] - self._means[j]) / self._stds[j] for j in range(len(row))]
            for row in data
        ]


class MinMaxScaler(DataPreprocessor):
    """Scale features to [0, 1] range."""

    def __init__(self):
        self._mins = None
        self._ranges = None

    def fit(self, data: list) -> 'MinMaxScaler':
        n_features = len(data[0])
        self._mins = []
        self._ranges = []
        for j in range(n_features):
            col = [row[j] for row in data]
            min_val = min(col)
            self._mins.append(min_val)
            self._ranges.append(max(col) - min_val if max(col) != min_val else 1.0)
        return self

    def transform(self, data: list) -> list:
        return [
            [(row[j] - self._mins[j]) / self._ranges[j] for j in range(len(row))]
            for row in data
        ]


# The user doesn't need to know the internals — just the interface
scaler = StandardScaler()
scaled_data = scaler.fit_transform([[10, 200], [20, 400], [30, 600]])
print(scaled_data)
# [[-1.224744871391589, -1.224744871391589],
#  [0.0, 0.0],
#  [1.224744871391589, 1.224744871391589]]
```

### How abstraction appears in real ML libraries

```python
# Scikit-learn — the user never sees the math inside
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler", StandardScaler()),   # Abstract: "normalize my data"
    ("pca", PCA(n_components=2))    # Abstract: "reduce to 2 dimensions"
])
X_transformed = pipe.fit_transform(X)  # Two complex ops, one clean call
```

---

## 6. Composition over Inheritance

In ML, composition (has-a relationship) is often preferred over deep inheritance hierarchies.

```python
class MLPipeline:
    """A machine learning pipeline composed of independent components."""

    def __init__(self, preprocessor, model, evaluator=None):
        self.preprocessor = preprocessor  # has-a
        self.model = model                # has-a
        self.evaluator = evaluator

    def run(self, X_train, y_train, X_test, y_test):
        print(f"Running pipeline: {self.model.name}")
        print("-" * 40)

        # Step 1: Preprocess
        print("  [1/4] Fitting preprocessor...")
        X_train_proc = self.preprocessor.fit_transform(X_train)
        X_test_proc = self.preprocessor.transform(X_test)

        # Step 2: Train
        print("  [2/4] Training model...")
        self.model.fit(X_train_proc, y_train)

        # Step 3: Predict
        print("  [3/4] Generating predictions...")
        predictions = self.model.predict(X_test_proc)

        # Step 4: Evaluate
        print("  [4/4] Evaluating...")
        if self.evaluator:
            results = self.evaluator(predictions, y_test)
            print(f"  Results: {results}")
            return results

        accuracy = sum(1 for p, t in zip(predictions, y_test) if p == t) / len(y_test)
        print(f"  Accuracy: {accuracy:.2%}")
        return accuracy

# Compose the pipeline
pipeline = MLPipeline(
    preprocessor=StandardScaler(),
    model=KNearestNeighbors(k=5)
)
pipeline.run(X_train, y_train, X_test, y_test)
```

---

## 7. Property Decorators

Properties let you add logic to attribute access while keeping a clean interface.

```python
class TrainingHistory:
    """Track and analyze training metrics over epochs."""

    def __init__(self):
        self._losses = []
        self._accuracies = []
        self._val_losses = []
        self._val_accuracies = []

    def log_epoch(self, loss, accuracy, val_loss=None, val_accuracy=None):
        self._losses.append(loss)
        self._accuracies.append(accuracy)
        self._val_losses.append(val_loss)
        self._val_accuracies.append(val_accuracy)

    @property
    def best_epoch(self) -> int:
        """The epoch with the lowest validation loss."""
        valid = [(i, l) for i, l in enumerate(self._val_losses) if l is not None]
        if not valid:
            valid = [(i, l) for i, l in enumerate(self._losses)]
        return min(valid, key=lambda x: x[1])[0]

    @property
    def is_overfitting(self) -> bool:
        """Check if the model is overfitting."""
        if len(self._val_losses) < 3:
            return False
        # Val loss increased for last 3 epochs while train loss decreased
        recent_val = self._val_losses[-3:]
        recent_train = self._losses[-3:]
        val_increasing = all(recent_val[i] < recent_val[i + 1]
                            for i in range(len(recent_val) - 1))
        train_decreasing = all(recent_train[i] > recent_train[i + 1]
                              for i in range(len(recent_train) - 1))
        return val_increasing and train_decreasing

    @property
    def total_epochs(self) -> int:
        return len(self._losses)

    @property
    def final_train_accuracy(self) -> float:
        return self._accuracies[-1] if self._accuracies else 0.0

history = TrainingHistory()
history.log_epoch(0.9, 0.5, 0.85, 0.48)
history.log_epoch(0.6, 0.7, 0.62, 0.68)
history.log_epoch(0.4, 0.8, 0.55, 0.72)
history.log_epoch(0.3, 0.85, 0.50, 0.75)
history.log_epoch(0.2, 0.90, 0.52, 0.74)  # Overfitting starts

print(f"Best epoch: {history.best_epoch}")           # 4 (index)
print(f"Total epochs: {history.total_epochs}")       # 5
print(f"Overfitting: {history.is_overfitting}")      # True
```

---

## 8. Magic Methods (Dunder Methods)

```python
class ConfusionMatrix:
    """Confusion matrix with rich Python integration."""

    def __init__(self, classes: list):
        self.classes = classes
        n = len(classes)
        self._matrix = [[0] * n for _ in range(n)]
        self._class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def add(self, true_label, pred_label):
        i = self._class_to_idx[true_label]
        j = self._class_to_idx[pred_label]
        self._matrix[i][j] += 1

    def __getitem__(self, key):
        """Access via cm[true, pred]."""
        true_label, pred_label = key
        i = self._class_to_idx[true_label]
        j = self._class_to_idx[pred_label]
        return self._matrix[i][j]

    def __str__(self):
        """Pretty-print the matrix."""
        width = max(len(c) for c in self.classes) + 2
        header = " " * width + "".join(f"{c:>{width}}" for c in self.classes)
        rows = [header]
        for i, cls in enumerate(self.classes):
            row = f"{cls:>{width}}" + "".join(f"{self._matrix[i][j]:>{width}}"
                                               for j in range(len(self.classes)))
            rows.append(row)
        return "\n".join(rows)

    def __len__(self):
        """Number of classes."""
        return len(self.classes)

    def accuracy(self) -> float:
        total = sum(self._matrix[i][j] for i in range(len(self)) for j in range(len(self)))
        correct = sum(self._matrix[i][i] for i in range(len(self)))
        return correct / total if total > 0 else 0.0

    def precision_recall_f1(self, label) -> dict:
        idx = self._class_to_idx[label]
        tp = self._matrix[idx][idx]
        fp = sum(self._matrix[i][idx] for i in range(len(self))) - tp
        fn = sum(self._matrix[idx][j] for j in range(len(self))) - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

# Usage
cm = ConfusionMatrix(["cat", "dog", "bird"])
for t, p in zip(
    ["cat", "dog", "cat", "bird", "dog", "cat", "bird", "dog", "cat", "bird"],
    ["cat", "dog", "dog",  "bird", "cat",  "cat", "bird", "dog", "cat", "dog"]
):
    cm.add(t, p)

print(cm)
print(f"Accuracy: {cm.accuracy():.2%}")
print(f"Cat metrics: {cm.precision_recall_f1('cat')}")
print(f"cm['cat', 'dog'] = {cm['cat', 'dog']}")  # 1
```

---

## 9. Exercises

### Exercise 1: Build a Linear Regression Class

Implement a simple linear regression using OOP with gradient descent.

```python
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        n_samples, n_features = len(X), len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.n_iter):
            # Forward pass
            predictions = [sum(w * x_j for w, x_j in zip(self.weights, xi)) + self.bias
                          for xi in X]

            # Compute MSE loss
            error = [p - t for p, t in zip(predictions, y)]
            mse = sum(e ** 2 for e in error) / (2 * n_samples)
            self.loss_history.append(mse)

            # Gradients
            dw = [sum(e * xi[j] for e, xi in zip(error, X)) / n_samples
                  for j in range(n_features)]
            db = sum(error) / n_samples

            # Update
            self.weights = [w - self.lr * g for w, g in zip(self.weights, dw)]
            self.bias -= self.lr * db

    def predict(self, X):
        return [sum(w * x_j for w, x_j in zip(self.weights, xi)) + self.bias
                for xi in X]

# Test
X_train = [[1], [2], [3], [4], [5]]
y_train = [2, 4, 6, 8, 10]

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X_train, y_train)
print(f"Weight: {model.weights}, Bias: {model.bias}")
print(f"Predict [6]: {model.predict([[6]])}")  # Should be ~12
```

### Exercise 2: Extend with Multiple Models

Create a `ModelComparator` class that takes a list of models (all implementing `fit` and `predict`) and runs cross-validation.

### Exercise 3: Magic Methods for Dataset

Create a `Dataset` class that supports:
- `len(dataset)` — number of samples
- `dataset[i]` — returns the i-th sample and label
- `dataset[2:5]` — slicing
- Iteration: `for sample, label in dataset:`

---

## Connections to Other ML Topics

| OOP Concept | ML Application |
|-------------|---------------|
| Inheritance | Custom estimators in scikit-learn, PyTorch `nn.Module` |
| Polymorphism | Swapping models in a pipeline seamlessly |
| Encapsulation | Hidden model parameters, controlled API |
| Abstraction | `pipeline.fit_transform()` hides all complexity |
| Composition | Building pipelines from independent components |
| Properties | Lazy computation of metrics, validation checks |
| Dunder methods | Custom containers for datasets, tensors |

---

## Key Takeaways

1. **Every ML framework is OOP** — understanding classes is prerequisite literacy.
2. **Inheritance** enables code reuse across model families (all classifiers share `fit`/`predict`).
3. **Polymorphism** lets you swap models in pipelines without changing pipeline code.
4. **Encapsulation** protects model internals while exposing a clean API.
5. **Abstraction** with ABCs enforces consistent interfaces across components.
6. **Composition** is preferred over deep inheritance — build pipelines from parts.
