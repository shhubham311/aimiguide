# Multi-class & Multi-modal Classifications

## Why This Matters

Real-world classification is rarely binary (spam/not spam). Products belong to dozens of categories, images contain multiple objects, and modern AI systems combine text, images, and audio simultaneously.

## Multi-class Classification

### Approaches

```
┌──────────────────────────────────────────────────────────┐
│ 1. One-vs-Rest (OvR)                                    │
│    Train K binary classifiers (one per class)            │
│    Predict: argmax of all K classifier scores            │
│    sklearn: OneVsRestClassifier                          │
│                                                           │
│ 2. One-vs-One (OvO)                                     │
│    Train K(K-1)/2 binary classifiers (all pairs)         │
│    Predict: majority vote                                │
│    sklearn: OneVsOneClassifier                           │
│                                                           │
│ 3. Multinomial (Native)                                  │
│    Single model with softmax output                      │
│    Most common, most efficient                           │
│    sklearn: LogisticRegression(multi_class='multinomial')│
└──────────────────────────────────────────────────────────┘
```

### Softmax Function

```
For K classes, the softmax converts logits to probabilities:

  P(y = k | x) = exp(z_k) / Σⱼ₌₁ᴷ exp(z_j)

Where z_k = w_k · x + b_k (logit for class k)
All probabilities sum to 1
```

### Implementation

```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

X, y = load_iris(return_X_y=True)  # 3 classes: setosa, versicolor, virginica

models = {
    "LogisticRegression": LogisticRegression(multi_class='multinomial', max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM (OvO)": SVC(decision_function_shape='ovo', random_state=42),
    "SVM (OvR)": SVC(decision_function_shape='ovr', random_state=42),
}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"{name:25s} Accuracy: {scores.mean():.4f} (±{scores.std():.4f})")

# Detailed evaluation
best = LogisticRegression(multi_class='multinomial', max_iter=1000)
best.fit(X, y)
y_pred = best.predict(X)
print(f"\nClassification Report:\n{classification_report(y, y_pred, target_names=load_iris().target_names)}")
```

## Handling Class Imbalance

```python
from sklearn.datasets import make_classification
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report

# Imbalanced dataset: 80% class 0, 20% class 1
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.8, 0.2],
                           random_state=42)

# Method 1: Class weights
weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weight_dict = dict(zip(np.unique(y), weights))
print(f"Class weights: {weight_dict}")

model = LogisticRegression(class_weight=weight_dict, max_iter=1000)
model.fit(X, y)
print(classification_report(y, model.predict(X)))

# Method 2: Oversampling (SMOTE)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)
print(f"Before SMOTE: {np.bincount(y)}")
print(f"After SMOTE:  {np.bincount(y_res)}")
```

## Multi-label Classification

Each sample can belong to **multiple classes** simultaneously.

```python
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression

# Movie genres: each movie can have multiple genres
movies = [
    (["action", "thriller"], [5.0, 3.0, 2.0]),
    (["comedy", "romance"], [3.0, 5.0, 4.0]),
    (["action", "sci-fi", "thriller"], [4.0, 2.0, 4.5]),
    (["drama", "romance"], [2.0, 4.0, 5.0]),
]

mlb = MultiLabelBinarizer()
y_encoded = mlb.fit_transform([labels for labels, _ in movies])
print(f"Classes: {mlb.classes_}")
print(f"Encoded:\n{y_encoded}")

# Multi-output classifier
X_features = np.array([features for _, features in movies])
clf = MultiOutputClassifier(LogisticRegression(max_iter=1000))
clf.fit(X_features, y_encoded)
```

## Multi-modal Classification

Combining information from multiple modalities (text + image, audio + video, etc.).

### Late Fusion (Concatenation)

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class MultiModalClassifier:
    """Late fusion: concatenate features from multiple modalities."""
    
    def __init__(self, model=None):
        self.model = model or RandomForestClassifier(n_estimators=100)
    
    def fit(self, X_text, X_image, y):
        X_combined = np.hstack([X_text, X_image])
        self.model.fit(X_combined, y)
        return self
    
    def predict(self, X_text, X_image):
        X_combined = np.hstack([X_text, X_image])
        return self.model.predict(X_combined)
    
    def predict_proba(self, X_text, X_image):
        X_combined = np.hstack([X_text, X_image])
        return self.model.predict_proba(X_combined)

# Simulated: 50-dim text features + 30-dim image features
np.random.seed(42)
n_samples = 200
X_text = np.random.randn(n_samples, 50)
X_image = np.random.randn(n_samples, 30)
y = np.random.randint(0, 3, n_samples)

clf = MultiModalClassifier()
clf.fit(X_text, X_image, y)
print(f"Accuracy: {clf.score(X_text, X_image, y):.4f}")
```

### Early Fusion vs Late Fusion

```
EARLY FUSION:                     LATE FUSION:
  [Text] ──→ Embedding             [Text] ──→ Model₁ ──→ Score₁
  [Image] ──→ Embedding  ──→       [Image] ──→ Model₂ ──→ Score₂
       Concatenate    Joint Model       ──→ Combine scores
  [Audio] ──→ Embedding             [Audio] ──→ Model₃ ──→ Score₃

  Pros: Joint representation       Pros: Independent modality models
  Cons: Alignment challenges        Cons: No cross-modal interaction
```

## Hierarchical Classification

```python
from sklearn.tree import DecisionTreeClassifier

# Hierarchical labels
# Level 1: Vehicle / Animal
# Level 2: Car/Bike (under Vehicle), Dog/Cat (under Animal)

# Approach: Train separate classifiers for each level
def hierarchical_predict(text_features, level1_model, level2_models):
    level1 = level1_model.predict([text_features])[0]
    if level1 in level2_models:
        level2 = level2_models[level1].predict([text_features])[0]
        return f"{level1} > {level2}"
    return level1
```

## 🧠 Exercises

### Exercise 1: Build a multi-class classifier on the digits dataset (10 classes) and analyze per-class performance.

<details>
<summary>Solution</summary>

```python
from sklearn.datasets import load_digits
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report

X, y = load_digits(return_X_y=True)
model = SVC(kernel='rbf', gamma=0.001)
scores = cross_val_score(model, X, y, cv=5)
print(f"Accuracy: {scores.mean():.4f}")

model.fit(X, y)
print(classification_report(y, model.predict(X)))
```

</details>

### Exercise 2: Implement a weighted softmax that handles class imbalance.

<details>
<summary>Solution</summary>

```python
import numpy as np

def weighted_softmax(logits, class_weights):
    weighted_logits = logits + np.log(class_weights)
    exp_logits = np.exp(weighted_logits - np.max(weighted_logits))
    return exp_logits / exp_logits.sum()

logits = np.array([2.0, 1.0, 0.5])
weights = np.array([0.2, 0.3, 0.5])  # Under-represented class gets higher weight
print(weighted_softmax(logits, weights))
```

</details>
