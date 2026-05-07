# Evaluation Metrics for Classification & Regression

## Why This Topic Matters

You cannot improve what you cannot measure. In machine learning, choosing the wrong evaluation metric can lead you to believe your model is performing well when it is actually failing catastrophically. A spam classifier with 99% accuracy might be useless if it lets through every single phishing email. A medical test with 95% accuracy might miss the one cancer case that matters most. Evaluation metrics are the lens through which you judge model quality, compare algorithms, and make deployment decisions. Understanding them deeply is what separates a model trainer from a machine learning engineer.

This lesson covers the two families of metrics—**classification metrics** and **regression metrics**—with mathematical formulations, visual intuition, and practical sklearn implementations on real datasets.

---

## Part 1: Classification Metrics

### The Confusion Matrix

Every classification metric starts with the **confusion matrix**. For a binary classification problem (Positive vs. Negative):

```
                         Predicted
                    Positive    Negative
                ┌────────────┬────────────┐
  Actual  Pos   │    TP      │    FN      │
                │ (True Pos) │ (False Neg)│
                ├────────────┼────────────┤
          Neg   │    FP      │    TN      │
                │ (False Pos)│ (True Neg) │
                └────────────┴────────────┘

TP (True Positive):   Correctly predicted positive
TN (True Negative):   Correctly predicted negative
FP (False Positive):  Predicted positive, actually negative (Type I Error)
FN (False Negative):  Predicted negative, actually positive (Type II Error)
```

### Visual: What Each Cell Means

```
Email Spam Example:

                Predicted Spam    Predicted Not Spam
                ┌──────────┬──────────────┐
Actual Spam     │    85     │     15       │  (100 spam emails)
                │    TP     │     FN       │
                ├──────────┼──────────────┤
Actual Not Spam │     5     │    895       │  (900 legitimate emails)
                │    FP     │     TN       │
                └──────────┴──────────────┘

We caught 85/100 spam emails (Recall = 0.85)
But we falsely flagged 5 legitimate emails (FP = 5)
```

---

### Metric 1: Accuracy

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**When to use**: Balanced classes (roughly equal positive and negative examples).

**When NOT to use**: Imbalanced datasets. A dataset with 95% negative class and 5% positive class can achieve 95% accuracy by simply predicting "negative" for everything.

```python
from sklearn.metrics import accuracy_score
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 0, 0, 1, 1]
print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
# Accuracy: 0.8000
```

### Metric 2: Precision

```
Precision = TP / (TP + FP)
```

**Meaning**: Of all the examples predicted as positive, how many were actually positive? Focuses on **minimizing false positives**.

**When it matters**: Spam filtering (don't send legitimate emails to spam folder), criminal sentencing (don't falsely convict), medical screening follow-ups (don't subject healthy people to invasive tests).

### Metric 3: Recall (Sensitivity / True Positive Rate)

```
Recall = TP / (TP + FN)
```

**Meaning**: Of all the actually positive examples, how many did we correctly identify? Focuses on **minimizing false negatives**.

**When it matters**: Cancer screening (don't miss any tumors), fraud detection (don't let fraud through), fault detection in manufacturing (don't ship defective products).

### Precision-Recall Tradeoff

```
Precision and Recall are in tension:

High Precision, Low Recall:
    "I only predict positive when I'm VERY sure"
    → Few false positives, but miss many true positives

High Recall, Low Precision:
    "I predict positive for anything that might be positive"
    → Catch most positives, but with many false positives

        Precision
            │ ●
            │  ●
            │   ●───────●
            │            ╲
            │             ╲───●
            │                  ●
            └──────────────────── Recall

    You cannot maximize both simultaneously — you must choose
    based on the business cost of FP vs. FN.
```

### Metric 4: F1 Score

The F1 score is the **harmonic mean** of precision and recall:

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)
```

Unlike the arithmetic mean, the harmonic mean heavily penalizes extreme imbalances. If either precision or recall is near zero, F1 will be near zero.

```
Example: Precision = 0.90, Recall = 0.10
    Arithmetic mean: (0.90 + 0.10) / 2 = 0.50  ← looks OK
    Harmonic mean:   2(0.90)(0.10)/(0.90+0.10) = 0.18  ← reveals poor performance

    F1 correctly tells you: this model is not good!
```

**When to use**: When you need a single number to compare models and the classes are imbalanced.

### Metric 5: F-beta Score

A generalization of F1 that lets you weight precision or recall more heavily:

```
F_β = (1 + β²) · (Precision · Recall) / (β² · Precision + Recall)
```

- `β = 1`: Standard F1 (equal weight)
- `β = 2`: Weighs recall 2x more than precision (use when FNs are costly)
- `β = 0.5`: Weighs precision 2x more than recall (use when FPs are costly)

### Metric 6: ROC-AUC

The **Receiver Operating Characteristic (ROC)** curve plots the True Positive Rate (Recall) against the False Positive Rate at various threshold settings:

```
TPR (Recall) = TP / (TP + FN)
FPR           = FP / (FP + TN)
```

```
ROC Curve:
    TPR
    1.0 │          ╱─────────────  (Perfect classifier)
        │        ╱
    0.8 │      ╱   ← Our model's ROC curve
        │    ╱  ╱
    0.6 │  ╱  ╱
        │╱  ╱
    0.4 │  ╱
        │╱
    0.2 │
        │──────────────────────────
    0.0 └────────────────────────── FPR
        0.0  0.2  0.4  0.6  0.8  1.0

    AUC = Area Under the Curve
    AUC = 1.0: Perfect
    AUC = 0.5: Random guessing
    AUC < 0.5: Worse than random (model is inverted)

    Diagonal line = random classifier
```

**AUC interpretation**: The probability that a randomly chosen positive example is ranked higher than a randomly chosen negative example by the model.

### All Metrics at a Glance

```
┌───────────────┬──────────────────────────┬──────────────────────────┐
│ Metric        │ Formula                  │ Best For                 │
├───────────────┼──────────────────────────┼──────────────────────────┤
│ Accuracy      │ (TP+TN)/(TP+TN+FP+FN)    │ Balanced classes         │
│ Precision     │ TP/(TP+FP)               │ Minimize false positives │
│ Recall        │ TP/(TP+FN)               │ Minimize false negatives │
│ F1            │ 2PR/(P+R)                │ Balanced class concern   │
│ ROC-AUC       │ Area under ROC curve     │ Threshold-independent    │
│ Specificity   │ TN/(TN+FP)               │ Correctly identify negs  │
└───────────────┴──────────────────────────┴──────────────────────────┘
```

---

## Python Implementation: Classification Metrics on Breast Cancer Data

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
import numpy as np

# Load data (binary classification: malignant vs. benign)
data = load_breast_cancer()
X, y = data.data, data.target  # 0=malignant, 1=benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]  # Probabilities for ROC-AUC

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(f"              Predicted")
print(f"              Neg   Pos")
print(f"Actual Neg    {cm[0,0]:4d}  {cm[0,1]:4d}")
print(f"Actual Pos    {cm[1,0]:4d}  {cm[1,1]:4d}")
print()

# All metrics at once
print("=" * 50)
print("CLASSIFICATION METRICS")
print("=" * 50)
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")

# Full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

# ROC Curve data
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
print(f"\nROC Curve sample points (threshold → FPR, TPR):")
for i in range(0, len(thresholds), len(thresholds) // 5):
    print(f"  threshold={thresholds[i]:.3f}: FPR={fpr[i]:.4f}, TPR={tpr[i]:.4f}")
```

**Expected Output:**
```
Confusion Matrix:
              Predicted
              Neg   Pos
Actual Neg      62     1
Actual Pos       3   105

==================================================
CLASSIFICATION METRICS
==================================================
Accuracy:  0.9766
Precision: 0.9906
Recall:    0.9722
F1 Score:  0.9813
ROC-AUC:   0.9947

Classification Report:
              precision    recall  f1-score   support

   Malignant       0.95      0.98      0.97        63
      Benign       0.99      0.97      0.98       108

    accuracy                           0.98       171
   macro avg       0.97      0.98      0.97       171
weighted avg       0.98      0.98      0.98       171
```

---

## Part 2: Regression Metrics

### The Setup

For regression, we compare predicted values `ŷᵢ` with actual values `yᵢ`. The goal is to quantify how close the predictions are to the truth.

```
Visual: Predictions vs. Actual

y (Actual)
│         *                  Perfect predictions
│      *  /                  lie on the diagonal.
│   *  / /
│  / / / *     *    ← good predictions cluster near line
│ / / /       /
│/___/_______/______ ŷ (Predicted)

    Points far from diagonal = large errors
```

### Metric 1: Mean Absolute Error (MAE)

```
MAE = (1/n) · Σ |yᵢ - ŷᵢ|
```

**Properties:**
- Same units as the target variable (interpretable!)
- Robust to outliers (errors are not squared)
- Linear penalty for each error unit

**Interpretation**: On average, predictions are off by `MAE` units.

### Metric 2: Mean Squared Error (MSE)

```
MSE = (1/n) · Σ (yᵢ - ŷᵢ)²
```

**Properties:**
- Penalizes large errors more heavily (quadratic penalty)
- Different units than target (units²) — less interpretable
- The standard loss function for linear regression
- Differentiable everywhere (used in gradient descent)

### Metric 3: Root Mean Squared Error (RMSE)

```
RMSE = √MSE = √[(1/n) · Σ (yᵢ - ŷᵢ)²]
```

**Properties:**
- Same units as target variable (more interpretable than MSE)
- Still penalizes large errors more than MAE
- Most commonly reported metric in practice

### Comparing MAE vs. RMSE

```
    Error Magnitude
    │
    │        RMSE penalty
    │       ╱
    │      ╱  (grows quadratically)
    │     ╱
    │────╱────── MAE penalty
    │   ╱        (grows linearly)
    │  ╱
    │ ╱
    └──────────────── Error size

    Example: Errors = [1, 1, 1, 1, 10]
    MAE  = (1+1+1+1+10)/5 = 2.8   ← dominated by the outlier less
    MSE  = (1+1+1+1+100)/5 = 20.8  ← outlier penalized heavily
    RMSE = √20.8 = 4.56

    If outliers are genuine data points: RMSE is appropriate
    If outliers are noise/measurement errors: MAE is more robust
```

### Metric 4: R² (Coefficient of Determination)

```
R² = 1 - (SS_res / SS_tot)

where:
    SS_res = Σ (yᵢ - ŷᵢ)²        (residual sum of squares)
    SS_tot = Σ (yᵢ - ȳ)²         (total sum of squares)
    ȳ = mean(y)                    (mean of actual values)
```

**Interpretation**:
- `R² = 1.0`: Perfect predictions
- `R² = 0.0`: Model predicts no better than the mean
- `R² < 0.0`: Model is worse than predicting the mean

```
R² represents the proportion of variance in y explained by the model:

    Total variance in y:     ████████████████████  100%
    Variance explained:      ████████████████      80%  → R² = 0.80
    Unexplained (residual):  █████                 20%
```

### Metric 5: Adjusted R²

R² always increases (or stays the same) when you add more features, even useless ones. **Adjusted R²** penalizes for adding features that don't improve the model:

```
R²_adj = 1 - [(1 - R²) · (n - 1) / (n - p - 1)]
```

where:
- `n` = number of samples
- `p` = number of features

**Rule**: If adding a feature doesn't improve predictions enough to overcome the penalty, Adjusted R² decreases.

### Metric 6: MAPE (Mean Absolute Percentage Error)

```
MAPE = (100/n) · Σ |yᵢ - ŷᵢ| / |yᵢ|
```

**Interpretation**: Average percentage that predictions deviate from actual values.

**Caveat**: Undefined when `yᵢ = 0`. Can be unstable for small values.

---

## Regression Metrics Comparison Table

```
┌─────────────┬──────────────┬──────────────────┬──────────────────┐
│   Metric    │  Units       │ Outlier Sens.    │  Interpretation  │
├─────────────┼──────────────┼──────────────────┼──────────────────┤
│ MAE         │ Same as y    │ Low              │ Avg. absolute    │
│             │              │                  │ error            │
│ MSE         │ y²           │ High             │ Avg. squared     │
│             │              │                  │ error            │
│ RMSE        │ Same as y    │ High             │ √ of avg.        │
│             │              │                  │ squared error    │
│ R²          │ Dimensionless│ Low              │ % variance       │
│             │ [0, 1]       │                  │ explained        │
│ R²_adj      │ Dimensionless│ Low              │ % variance       │
│             │              │                  │ explained (adj.) │
│ MAPE        │ Percentage   │ Medium           │ Avg. % error     │
└─────────────┴──────────────┴──────────────────┴──────────────────┘
```

---

## Python Implementation: Regression Metrics on California Housing

```python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    mean_absolute_percentage_error
)
import numpy as np

# Load dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Train multiple models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1)': Ridge(alpha=1.0),
    'Ridge (α=100)': Ridge(alpha=100.0),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
}

print("=" * 75)
print(f"{'Model':<20} {'MAE':>8} {'MSE':>10} {'RMSE':>8} {'R²':>8} {'MAPE%':>8}")
print("=" * 75)

results = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred) * 100

    results[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}
    print(f"{name:<20} {mae:>8.4f} {mse:>10.4f} {rmse:>8.4f} {r2:>8.4f} {mape:>7.2f}%")

print("=" * 75)
```

**Expected Output:**
```
===========================================================================
Model                   MAE       MSE     RMSE       R²    MAPE%
===========================================================================
Linear Regression     0.5332     0.5559   0.7456   0.5757   29.79%
Ridge (α=1)           0.5333     0.5559   0.7456   0.5757   29.80%
Ridge (α=100)         0.5516     0.5752   0.7584   0.5612   30.76%
Random Forest         0.3277     0.2554   0.5054   0.8051   17.87%
===========================================================================
```

---

## Multi-Class Classification Metrics

For problems with more than 2 classes, metrics are computed **per-class** and then aggregated:

### Strategies:
1. **Macro-average**: Average of per-class metrics (treats all classes equally)
2. **Weighted-average**: Average weighted by class support (number of true instances)
3. **Micro-average**: Aggregate all TP, FP, FN globally, then compute

```python
from sklearn.metrics import precision_score, recall_score, f1_score

y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2]
y_pred = [0, 1, 1, 0, 2, 2, 0, 1, 0]  # Some mistakes

print("Per-class strategies:")
print(f"  Macro Precision:    {precision_score(y_true, y_pred, average='macro'):.4f}")
print(f"  Weighted Precision: {precision_score(y_true, y_pred, average='weighted'):.4f}")
print(f"  Micro Precision:    {precision_score(y_true, y_pred, average='micro'):.4f}")

print(f"\n  Macro F1:    {f1_score(y_true, y_pred, average='macro'):.4f}")
print(f"  Weighted F1: {f1_score(y_true, y_pred, average='weighted'):.4f}")
```

---

## Exercises

### Exercise 1: Imbalanced Classification Challenge

A disease affects only 5% of patients. Build a classifier and compare accuracy vs. F1 score.

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Create imbalanced dataset: 95% class 0, 5% class 1
X, y = make_classification(
    n_samples=10000, n_features=20,
    weights=[0.95, 0.05], flip_y=0, random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Baseline: always predict majority class
dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
y_pred_dummy = dummy.predict(X_test)

print("Dummy Classifier (always predicts 0):")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_dummy):.4f}")
print(f"  F1 Score: {f1_score(y_test, y_pred_dummy):.4f}")

# Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  F1 Score: {f1_score(y_test, y_pred_rf):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))
```

### Solution 1:
```
Dummy Classifier (always predicts 0):
  Accuracy: 0.9500    ← Looks great but is useless!
  F1 Score: 0.0000    ← Reveals the truth: model predicts no positives

Random Forest:
  Accuracy: 0.9850
  F1 Score: 0.7071    ← Much better for the minority class

The dummy has 95% accuracy but F1=0 because it never predicts the
positive class. This demonstrates why accuracy is misleading for
imbalanced datasets.
```

### Exercise 2: Regression Model Selection

Compare three regression models on the diabetes dataset using MAE, RMSE, and R².

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

X, y = load_diabetes(return_X_y=True)

models = {
    'Linear': make_pipeline(StandardScaler(), LinearRegression()),
    'Ridge': make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    'Lasso': make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
    'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
}

for name, model in models.items():
    mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
    print(f"{name:15s}: MAE={mae_scores.mean():.3f} ± {mae_scores.std():.3f}, "
          f"R²={r2_scores.mean():.3f} ± {r2_scores.std():.3f}")
```

---

## Key Takeaways

1. **Accuracy** is misleading for imbalanced datasets; always report **precision, recall, and F1** as well.
2. **Precision** minimizes false positives; **recall** minimizes false negatives. Choose based on business costs.
3. **F1 score** is the harmonic mean of precision and recall — a single metric for imbalanced problems.
4. **ROC-AUC** is threshold-independent and measures ranking quality.
5. For regression, **MAE** is interpretable and robust; **RMSE** penalizes large errors; **R²** shows explained variance.
6. **Always use multiple metrics** — no single metric tells the whole story.

---

*Previous: [Linear Regression & Ridge Regression](./linear-regression.md) | Next: [Logistic Regression & Gradient Descent](./logistic-regression.md)*
