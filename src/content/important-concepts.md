# Important ML Concepts for Interviews

## Overview

This guide covers the highest-frequency ML concepts that appear in AI company interviews. Master these and you'll handle 80%+ of conceptual questions.

## 1. Bias-Variance Tradeoff

The fundamental tension in ML:

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias (Underfitting):         High Variance (Overfitting):
  - Model too simple                - Model too complex
  - Can't capture patterns          - Memorizes noise
  - High training error             - Low training, high test error
  - Solution: more complex model   - Solution: regularization, more data

     Error     Error
      |  /\      |  \
      | /  \     |   \
      |/    \    |    \
      |______\   |_____\___Training → Test error grows
      Low fit   High fit, low test
```

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import numpy as np

np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(80) * 0.3

for degree in [1, 3, 10, 20]:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree)),
        ('lr', LinearRegression())
    ])
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"Degree {degree:2d}: Train MSE = {-scores.mean():.4f} (±{scores.std():.4f})")
```

## 2. Regularization

```
┌──────────────────────────────────────────────────┐
│ RIDGE (L2):        LASSO (L1):                   │
│ Loss + λΣwᵢ²       Loss + λΣ|wᵢ|               │
│                                                   │
│ Shrinks all weights  Drives some weights to 0    │
│ Never zero           Feature selection built-in   │
│ Closed-form solution No closed-form              │
│ Use when: correlated    Use when: sparse          │
│ features              features expected           │
└──────────────────────────────────────────────────┘

Elastic Net = α × L1 + (1-α) × L2  (best of both!)
```

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=200, n_features=20, n_informative=5, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)

for name, model in [("Ridge", Ridge(alpha=1)), ("Lasso", Lasso(alpha=0.1)), ("ElasticNet", ElasticNet(alpha=0.1))]:
    model.fit(X_train, y_train)
    print(f"{name:12s} R² = {model.score(X_test, y_test):.4f}, Non-zero coefs: {np.sum(np.abs(model.coef_) > 0.01)}")
```

## 3. Cross-Validation

```
K-Fold:                        Stratified K-Fold:
┌───┬───┬───┬───┬───┐         Preserves class ratio in each fold
│ T │ T │ T │ T │ V │         
├───┼───┼───┼───┼───┤         Important for imbalanced datasets
│ T │ T │ T │ V │ T │         
├───┼───┼───┼───┼───┤         
│ T │ T │ V │ T │ T │         
├───┼───┼───┼───┼───┤         
│ T │ V │ T │ T │ T │         
├───┼───┼───┼───┼───┤         
│ V │ T │ T │ T │ T │         
└───┴───┴───┴───┴───┘         
T = Train, V = Validate
```

## 4. Feature Engineering

```python
# Key techniques:
# 1. Missing value handling
# 2. Encoding (one-hot, label, target)
# 3. Scaling (standard, min-max, robust)
# 4. Binning/discretization
# 5. Polynomial features
# 6. Interaction features
# 7. Date/time features
# 8. Text features (TF-IDF, embeddings)

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
data = [[-10, 100], [0, 200], [10, 300], [1000, 400]]

print("Standard:", StandardScaler().fit_transform(data).round(2))
print("MinMax:  ", MinMaxScaler().fit_transform(data).round(2))
print("Robust:  ", RobustScaler().fit_transform(data).round(2))
```

## 5. Gradient Descent Variants

```
Batch GD:       Stochastic:    Mini-batch:
  Uses ALL data    Uses 1 point   Uses batch of m points
  Slow, stable     Fast, noisy    Best of both
  θ -= α∇J(θ,X)   θ -= α∇J(θ,xi) θ -= α∇J(θ,batch)
  
  Cost smooth      Cost jumpy      Cost moderate
```

## 6. Ensemble Methods

```
BAGGING (Parallel):              BOOSTING (Sequential):
  Train models independently       Train models sequentially
  Average predictions             Each fixes previous errors
  Reduces variance                Reduces bias
  Example: Random Forest          Example: AdaBoost, XGBoost

Stacking:                         Blending:
  Level 1: Multiple models         Similar to stacking but
  Level 2: Meta-learner on          uses a holdout set
           level 1 predictions      instead of cross-val
```

## 7. The ML Pipeline Checklist

```
1. Problem definition → Regression? Classification? Clustering?
2. Data collection → Sources, API, scraping
3. EDA → Distributions, correlations, missing values
4. Data cleaning → Imputation, outlier handling
5. Feature engineering → New features, transformations
6. Feature selection → Remove irrelevant features
7. Train/validation split → Stratified if needed
8. Model selection → Try multiple algorithms
9. Hyperparameter tuning → Grid/random/bayesian search
10. Cross-validation → Reliable performance estimate
11. Final evaluation → On held-out test set
12. Deployment → API, container, monitoring
```

## 8. Common Interview Questions

| Question | Key Points |
|----------|-----------|
| Why does ML fail? | Bad data, overfitting, wrong assumptions, data leakage |
| What is data leakage? | Test info leaking into training (e.g., preprocessing before split) |
| How to handle imbalance? | Resampling, class weights, SMOTE, focal loss |
| How to handle missing data? | Mean/median impute, model-based, drop if < 5% |
| What is the curse of dimensionality? | Distances lose meaning in high dimensions |
| When to use neural nets vs trees? | NN for unstructured data, trees for tabular |
| How to debug an ML model? | Check data, simplify model, check one sample, use SHAP |

## 🧠 Exercise: End-to-End ML Checklist

Given a dataset of customer transactions, walk through the complete ML pipeline to predict churn. Include: data loading, EDA, preprocessing, model training, evaluation, and interpretation.

<details>
<summary>Solution approach</summary>

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
import numpy as np

# 1. Generate data (simulating customer churn)
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                           n_redundant=5, weights=[0.7, 0.3], random_state=42)

# 2. Split (stratified for imbalance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    stratify=y, random_state=42)

# 3. Build pipelines with preprocessing
pipelines = {
    'LogisticRegression': Pipeline([('scaler', StandardScaler()),
                                   ('model', LogisticRegression(max_iter=1000))]),
    'RandomForest': Pipeline([('model', RandomForestClassifier(random_state=42))]),
    'GradientBoosting': Pipeline([('model', GradientBoostingClassifier(random_state=42))]),
}

# 4. Cross-validation
for name, pipe in pipelines.items():
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='roc_auc')
    print(f"{name:20s} AUC: {scores.mean():.4f} (±{scores.std():.4f})")

# 5. Best model → hyperparameter tuning
param_grid = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [3, 5, 7]}
grid = GridSearchCV(pipelines['RandomForest'], param_grid, cv=5, scoring='roc_auc')
grid.fit(X_train, y_train)
print(f"\nBest params: {grid.best_params_}")
print(f"Best AUC: {grid.best_score_:.4f}")

# 6. Final evaluation
y_pred = grid.predict(X_test)
print(f"\nTest AUC: {roc_auc_score(y_test, grid.predict_proba(X_test)[:,1]):.4f}")
print(classification_report(y_test, y_pred))
```

</details>
