# Spam/Outlier Detection & Classification

## Why This Matters

Spam detection and anomaly/outlier detection are two sides of the same coin: **finding patterns that deviate from the norm**. Applications range from email filtering to fraud detection, manufacturing quality control, and ML model monitoring.

## 1. Spam Detection

### Classic Approach: Naive Bayes

```
P(spam | words) = P(words | spam) × P(spam) / P(words)

The "naive" assumption: words are conditionally independent given the class.

Spam probability = Π P(wordᵢ | spam) × P(spam)
Ham probability  = Π P(wordᵢ | ham) × P(ham)
Classify as spam if spam_prob > ham_prob
```

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

# Spam dataset
emails = [
    ("Free money now!!! Click here to claim your prize", "spam"),
    ("Meeting at 3pm tomorrow with the team", "ham"),
    ("URGENT: Your account has been compromised. Act now!", "spam"),
    ("Can you review the document I sent yesterday?", "ham"),
    ("Congratulations! You've won a $1000 gift card!", "spam"),
    ("The project deadline is next Friday", "ham"),
    ("Limited time offer! Buy one get one free!", "spam"),
    ("Lunch today? I'm thinking Italian food", "ham"),
    ("Make fast cash from home. No experience needed!", "spam"),
    ("Your package has been shipped and will arrive Monday", "ham"),
]

texts, labels = zip(*emails)

# Feature extraction
vectorizer = TfidfVectorizer(stop_words='english', max_features=100)
X = vectorizer.fit_transform(texts)

# Train
model = MultinomialNB()
scores = cross_val_score(model, X, labels, cv=3)
print(f"Accuracy: {scores.mean():.4f}")

model.fit(X, labels)
print(classification_report(labels, model.predict(X), target_names=["ham", "spam"]))

# Predict new email
new_email = vectorizer.transform(["You won't believe this amazing offer!"])
print(f"Prediction: {model.predict(new_email)[0]}")
```

## 2. Imbalanced Classification

Spam detection is inherently imbalanced: spam is a small fraction of all emails.

```
Dataset: 95% ham, 5% spam
Naive accuracy: 95% (by always predicting "ham"!)
→ Need better metrics: Precision, Recall, F1

┌──────────────────────────────────────────────────┐
│  Confusion Matrix                                │
│                                                   │
│                 Predicted                          │
│              Ham    Spam                           │
│  Actual  Ham  TN     FP                           │
│          Spam FN     TP                           │
│                                                   │
│  Precision = TP / (TP + FP)  "Of predicted spam,  │
│            how many actually are?"                 │
│                                                   │
│  Recall    = TP / (TP + FN)  "Of actual spam,     │
│            how many did we catch?"                 │
│                                                   │
│  F1        = 2 × (P × R) / (P + R)              │
└──────────────────────────────────────────────────┘
```

### Handling Imbalance

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np

# Highly imbalanced dataset
X, y = make_classification(n_samples=5000, n_features=20,
                           n_informative=10, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print(f"Class distribution: {np.bincount(y)}")
print(f"Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")

# Method 1: class_weight='balanced'
model_balanced = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
model_balanced.fit(X_train, y_train)
print(f"\nBalanced weights:\n{classification_report(y_test, model_balanced.predict(X_test))}")

# Method 2: scale_pos_weight (ratio of neg/pos)
neg_count = np.sum(y_train == 0)
pos_count = np.sum(y_train == 1)
scale = neg_count / pos_count
model_scaled = RandomForestClassifier(n_estimators=100, class_weight={0: 1, 1: scale}, random_state=42)
model_scaled.fit(X_train, y_train)
print(f"Scaled weights:\n{classification_report(y_test, model_scaled.predict(X_test))}")
```

## 3. Anomaly/Outlier Detection

### Isolation Forest

```
Idea: Anomalies are "few and different" → easier to isolate.
Randomly partition data; anomalies need fewer splits to isolate.

IsolationForest builds random decision trees:
  - Normal points: many splits needed (deep in tree)
  - Anomalies: few splits needed (shallow in tree)

Anomaly Score: avg path length across all trees
  Shorter path → more anomalous
```

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

# Normal data with a few outliers
np.random.seed(42)
normal = np.random.normal(0, 1, (200, 2))
outliers = np.array([[5, 5], [-4, 6], [6, -5], [-6, -6], [0, 8]])
data = np.vstack([normal, outliers])

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Fit Isolation Forest
iso_forest = IsolationForest(contamination=0.03, random_state=42)
predictions = iso_forest.fit_predict(data_scaled)

# -1 = outlier, 1 = normal
outlier_mask = predictions == -1
print(f"Detected {outlier_mask.sum()} outliers out of {len(data)} points")

for i in range(len(data)):
    label = "OUTLIER" if outlier_mask[i] else "normal"
    print(f"  {data[i]} → {label}")
```

### One-Class SVM

```python
from sklearn.svm import OneClassSVM

# Train only on NORMAL data
normal_data = normal[:150]  # Use first 150 normal points
normal_scaled = scaler.transform(normal_data)

oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
oc_svm.fit(normal_scaled)

# Predict on all data
predictions = oc_svm.predict(data_scaled)
# -1 = outlier, 1 = normal
print(f"One-Class SVM detected {(predictions == -1).sum()} outliers")
```

### Statistical Method: Z-Score

```python
def detect_outliers_zscore(data, threshold=3):
    """Detect outliers using Z-score method."""
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    z_scores = np.abs((data - mean) / std)
    outlier_mask = np.any(z_scores > threshold, axis=1)
    return outlier_mask, z_scores

outlier_mask, z_scores = detect_outliers_zscore(data_scaled, threshold=2.5)
print(f"Z-score method detected {outlier_mask.sum()} outliers")
```

## 4. Model Monitoring: Detecting Data Drift

```python
from scipy.stats import ks_2samp

def detect_data_drift(reference_data, new_data, threshold=0.05):
    """
    Kolmogorov-Smirnov test for each feature.
    Returns features with significant drift.
    """
    drifted_features = []
    for i in range(reference_data.shape[1]):
        statistic, p_value = ks_2samp(reference_data[:, i], new_data[:, i])
        if p_value < threshold:
            drifted_features.append({
                'feature': f'feature_{i}',
                'statistic': statistic,
                'p_value': p_value
            })
    return drifted_features

# Simulate: reference data vs drifted data
ref_data = np.random.normal(0, 1, (1000, 5))
new_data = np.random.normal(0, 1, (1000, 5))
new_data[:, 0] = np.random.normal(1, 1, 1000)  # Drift in feature 0!

drift = detect_data_drift(ref_data, new_data)
print(f"Features with drift: {len(drift)}")
for f in drift:
    print(f"  {f['feature']}: KS={f['statistic']:.4f}, p={f['p_value']:.6f}")
```

## 5. Precision-Recall Tradeoff

```python
from sklearn.metrics import precision_recall_curve, auc
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(class_weight='balanced', max_iter=1000)
model.fit(X_train, y_train)
y_scores = model.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

# Find optimal threshold (maximize F1)
f1_scores = 2 * precision * recall / (precision + recall + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"PR AUC: {pr_auc:.4f}")
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"At optimal threshold: P={precision[optimal_idx]:.4f}, R={recall[optimal_idx]:.4f}, F1={f1_scores[optimal_idx]:.4f}")
```

## 🧠 Exercises

### Exercise 1: Build a spam filter using TF-IDF + Logistic Regression. Evaluate precision, recall, and F1 specifically for the spam class.

<details>
<summary>Solution</summary>

```python
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts)
model = LogisticRegression(class_weight='balanced', max_iter=1000)
y_pred = cross_val_predict(model, X, labels, cv=4)
print(classification_report(labels, y_pred, target_names=["ham", "spam"]))
```

</details>

### Exercise 2: Use Isolation Forest to detect anomalies in the California Housing dataset. Identify which features contribute most to anomalies.

<details>
<summary>Solution approach</summary>

```python
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import IsolationForest

housing = fetch_california_housing()
iso = IsolationForest(contamination=0.05, random_state=42)
iso.fit(housing.data)
anomalies = iso.predict(housing.data) == -1
anomaly_data = housing.data[anomalies]
normal_data = housing.data[~anomalies]
# Compare mean of each feature for anomalies vs normal
for i, name in enumerate(housing.feature_names):
    print(f"{name:15s}: anomaly_mean={anomaly_data[:,i].mean():.2f}, normal_mean={normal_data[:,i].mean():.2f}")
```

</details>
