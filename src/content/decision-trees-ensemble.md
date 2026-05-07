# Decision Trees & Ensemble Learning (Bagging & Boosting)

## 1. Why This Topic Matters

Decision trees are among the most intuitive and widely used machine learning models. They form the backbone of powerful ensemble methods like **Random Forests** and **Gradient Boosting Machines (GBMs)**, which dominate tabular data competitions (Kaggle), credit scoring, medical diagnosis, and recommendation systems. Understanding trees is essential because they are the building blocks of XGBoost, LightGBM, and CatBoost — three of the most successful ML algorithms in production today.

---

## 2. Decision Trees: Core Concepts

### 2.1 Anatomy of a Decision Tree

A decision tree splits the feature space into regions through a sequence of yes/no questions.

```
                    [Feature x1 <= 3.2?]
                       /            \
                    Yes              No
                   /                    \
          [Feature x2 <= 1.5?]    [Feature x1 <= 7.8?]
              /         \            /          \
           Yes          No        Yes           No
           /              \       /               \
        [Class A]    [Class B]  [Class B]      [Class A]
```

- **Root Node:** The topmost node (first split).
- **Internal Nodes:** Decision points that split data.
- **Leaf Nodes:** Final predictions (class labels or values).
- **Branches:** Outcomes of a test (True/False).

### 2.2 Splitting Criteria for Classification

#### Gini Impurity

Measures the probability of misclassifying a randomly chosen element:

```
Gini(t) = 1 - Σ (p_i)^2
```

Where `p_i` is the proportion of samples of class `i` in node `t`. A Gini of 0 means the node is pure (all same class).

#### Entropy (Information Gain)

```
Entropy(t) = - Σ p_i * log2(p_i)
```

The **Information Gain** from a split is:

```
IG = Entropy(parent) - Σ (N_left / N) * Entropy(left) - Σ (N_right / N) * Entropy(right)
```

### 2.3 Splitting Criteria for Regression

For regression trees, the split criterion is typically the **Reduction in Variance** (MSE):

```
MSE(t) = (1/N) * Σ (y_i - y_bar)^2

where y_bar = (1/N) * Σ y_i  (mean of target values in node t)
```

The tree selects the split that minimizes the weighted MSE of the child nodes.

---

## 3. Building a Decision Tree (Algorithm)

The CART (Classification and Regression Trees) algorithm builds trees greedily:

```
Algorithm: BuildTree(data, max_depth)
    IF stopping_criterion_met(data):
        RETURN LeafNode(prediction = majority_class or mean)

    best_feature, best_threshold = find_best_split(data)

    left_data  = data[data[best_feature] <= best_threshold]
    right_data = data[data[best_feature] >  best_threshold]

    node = InternalNode(feature=best_feature, threshold=best_threshold)
    node.left  = BuildTree(left_data,  max_depth - 1)
    node.right = BuildTree(right_data, max_depth - 1)
    RETURN node
```

### 3.1 Regularization (Preventing Overfitting)

Unconstrained trees will grow until every leaf is pure — overfitting severely. Key hyperparameters:

| Parameter | Effect |
|-----------|--------|
| `max_depth` | Limits tree depth |
| `min_samples_split` | Min samples to allow a split |
| `min_samples_leaf` | Min samples in a leaf node |
| `max_features` | Max features considered per split |
| `ccp_alpha` | Cost-complexity pruning parameter |

**Cost-Complexity Pruning:**

```
R_alpha(T) = R(T) + alpha * |T|

R(T) = total misclassification rate of tree T
|T|  = number of terminal nodes
alpha = complexity parameter (>= 0)
```

---

## 4. Ensemble Methods: Why Combine Trees?

A single decision tree is often high-variance (unstable). Ensemble methods combine multiple trees to reduce variance and bias.

```
Single Tree:          High Variance, Low Bias
    ▲                    o   o   o
    │                  o       o
    │                    o   o
    └──────────► x

Ensemble (many trees):  Low Variance, Low Bias
    ▲                    ●
    │                  ●
    │                ●
    └──────────► x
```

### 4.1 Bagging (Bootstrap Aggregating)

**Key Idea:** Train multiple trees on different bootstrap samples, then average predictions.

```
Original Dataset (N samples)
  │
  ├──► Bootstrap Sample 1 ──► Tree 1 ──┐
  ├──► Bootstrap Sample 2 ──► Tree 2 ──┤
  ├──► Bootstrap Sample 3 ──► Tree 3 ──┼──► Aggregation (Majority Vote / Average)
  ├──► ...                              │
  └──► Bootstrap Sample B ──► Tree B ──┘

Each bootstrap sample: N samples drawn WITH replacement
~63.2% unique samples per bag (1 - 1/e ≈ 0.632)
```

**Variance Reduction:**

If we have B independent trees each with variance σ², the ensemble variance is:

```
Var(ensemble) = σ² / B + ρ * σ²

where:
  σ² = variance of individual tree
  B  = number of trees
  ρ  = correlation between trees (ρ < 1 for bagging)
```

### 4.2 Random Forest

Random Forest = Bagging + Feature Subsampling

At each split, only `sqrt(p)` features (for classification) or `p/3` (for regression) are considered:

```
Algorithm: RandomForest(data, B, mtry)
    predictions = []
    FOR b = 1 to B:
        sample = bootstrap(data)           # Random sample of rows
        tree = build_tree(sample,
                  max_features = mtry)     # Random subset of columns
        predictions.append(tree.predict(X))
    RETURN aggregate(predictions)           # Vote (class) or Mean (regression)
```

### 4.3 Boosting

**Key Idea:** Train trees sequentially, where each new tree corrects the errors of previous trees.

```
Weak Learner 1 ──► Errors 1 ──► Weak Learner 2 ──► Errors 2 ──► ... ──► Final Model

Step 1: Train Tree_1 on original data
Step 2: Compute residuals: r_i = y_i - Tree_1(x_i)
Step 3: Train Tree_2 on residuals r_i
Step 4: Update: F_2(x) = F_1(x) + η * Tree_2(x)
Step 5: Repeat for M rounds
Final: F_M(x) = Σ η * Tree_m(x)
```

**Mathematical Formulation (Gradient Boosting):**

```
F_0(x) = argmin_c Σ L(y_i, c)

For m = 1 to M:
    r_i^(m) = - [∂L(y_i, F(x_i)) / ∂F(x_i)]  at F = F_(m-1)   # Pseudo-residuals
    h_m(x) = fit_tree(r^(m), X)                                  # Weak learner
    γ_m    = argmin_γ Σ L(y_i, F_(m-1)(x_i) + γ * h_m(x_i))     # Line search
    F_m(x) = F_(m-1)(x) + η * γ_m * h_m(x)                     # Update

where:
  L    = loss function (MSE, log-loss, etc.)
  η    = learning rate (shrinkage, typically 0.01–0.3)
  γ_m  = optimal step size
```

### 4.4 Bagging vs Boosting Comparison

| Aspect | Bagging (Random Forest) | Boosting (XGBoost, etc.) |
|--------|------------------------|--------------------------|
| Training | Parallel | Sequential |
| Goal | Reduce variance | Reduce bias |
| Base learners | Deep trees | Shallow trees (stumps) |
| Overfitting risk | Lower | Higher (needs tuning) |
| Handles outliers | Better | Sensitive |
| Speed | Faster (parallel) | Slower (sequential) |

---

## 5. Python Implementation

### 5.1 Single Decision Tree

```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load real dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Single Decision Tree (unpruned — likely overfits)
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
print(f"Single Tree Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Typically: ~0.9474 — but train accuracy is 1.0 (overfitting!)

# Pruned Decision Tree
dt_pruned = DecisionTreeClassifier(
    max_depth=5,
    min_samples_leaf=5,
    random_state=42
)
dt_pruned.fit(X_train, y_train)
y_pred_pruned = dt_pruned.predict(X_test)
print(f"Pruned Tree Accuracy: {accuracy_score(y_test, y_pred_pruned):.4f}")
# Typically: ~0.9561 — better generalization

# Inspect the tree (first few splits)
tree_rules = export_text(dt_pruned, feature_names=list(data.feature_names))
print(tree_rules[:500])
```

### 5.2 Random Forest

```python
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Random Forest with 200 trees
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=2,
    max_features='sqrt',     # sqrt(p) features per split
    random_state=42,
    n_jobs=-1                # Use all cores
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
# Typically: ~0.9649

# Feature Importance
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
print("Top 5 Features:")
for i in range(5):
    print(f"  {data.feature_names[indices[i]]:30s} {importances[indices[i]]:.4f}")
```

### 5.3 Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=3,             # Shallow trees
    learning_rate=0.05,      # Low learning rate
    subsample=0.8,           # Stochastic gradient boosting
    random_state=42
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print(f"Gradient Boosting Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}")
# Typically: ~0.9649

# Learning curves: effect of n_estimators
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

train_errors, test_errors = [], []
for n in [10, 50, 100, 200, 300, 500]:
    model = GradientBoostingClassifier(
        n_estimators=n, max_depth=3, learning_rate=0.1, random_state=42
    )
    model.fit(X_train, y_train)
    train_errors.append(log_loss(y_train, model.predict_proba(X_train)[:, 1]))
    test_errors.append(log_loss(y_test, model.predict_proba(X_test)[:, 1]))

for n, te, ve in zip([10,50,100,200,300,500], train_errors, test_errors):
    print(f"  n={n:3d}  Train Loss: {te:.4f}  Test Loss: {ve:.4f}")
```

### 5.4 Regression Example

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X, y = housing.data, housing.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_reg.fit(X_train, y_train)
print(f"RF Regression R²: {rf_reg.score(X_test, y_test):.4f}")

# Gradient Boosting Regressor
gb_reg = GradientBoostingRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, random_state=42
)
gb_reg.fit(X_train, y_train)
print(f"GB Regression R²: {gb_reg.score(X_test, y_test):.4f}")
```

---

## 6. Real-World ML Connections

- **Credit Scoring:** Random Forests are used at major banks to assess loan default risk. Feature importance reveals which factors (income, credit history, debt ratio) matter most.
- **Medical Diagnosis:** Gradient Boosting models detect diseases from patient data. XGBoost is widely used for ICU mortality prediction and cancer classification.
- **Fraud Detection:** Ensemble methods detect anomalous transactions in real-time at companies like PayPal and Visa.
- **Kaggle Competitions:** XGBoost, LightGBM, and CatBoost have won the majority of structured data competitions since 2015.
- **Feature Selection:** Tree-based feature importance scores are used to identify the most predictive variables before building more complex models.

---

## 7. Key Hyperparameter Tuning Guide

```
Random Forest:
  n_estimators     : 100–500 (more is better, diminishing returns)
  max_depth        : 10–30 (or None)
  min_samples_leaf : 1–5
  max_features     : 'sqrt' (classification), 'log2', or 0.3*p

Gradient Boosting:
  n_estimators     : 100–1000 (use early_stopping)
  max_depth        : 3–8 (shallow trees!)
  learning_rate    : 0.01–0.3 (lower needs more trees)
  subsample        : 0.5–1.0 (stochastic GB)
  min_child_weight : 1–10 (controls overfitting)
```

**Trade-off:** Low learning rate + more trees = better generalization but slower training.

---

## 8. Exercises

### Exercise 1: Decision Tree Depth Analysis
Train Decision Trees with `max_depth` values of 1, 3, 5, 10, None on the breast cancer dataset. Plot train and test accuracy vs depth. Identify the optimal depth.

### Exercise 2: Out-of-Bag Error
Random Forests provide free validation via OOB samples. Use `oob_score=True` and compare OOB score with cross-validation score.

### Exercise 3: Feature Importance Comparison
Compare feature importances from a single tree, Random Forest, and Gradient Boosting. Are the top features the same? Why or why not?

### Exercise 4: Learning Rate Sweep
Train Gradient Boosting with learning rates [0.01, 0.05, 0.1, 0.3, 0.5] and 500 trees each. Plot test log-loss vs iteration for each rate.

### Solutions

**Solution 1:**
```python
depths = [1, 3, 5, 10, None]
train_accs, test_accs = [], []
for d in depths:
    dt = DecisionTreeClassifier(max_depth=d, random_state=42).fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
    test_accs.append(accuracy_score(y_test, dt.predict(X_test)))
    print(f"Depth={str(d):4s}  Train: {train_accs[-1]:.4f}  Test: {test_accs[-1]:.4f}")
# Optimal depth is typically 3–5 (beyond that, overfitting increases)
```

**Solution 2:**
```python
rf = RandomForestClassifier(n_estimators=200, oob_score=True, random_state=42)
rf.fit(X_train, y_train)
print(f"OOB Score: {rf.oob_score_:.4f}")
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(rf, X_train, y_train, cv=5).mean()
print(f"CV Score:  {cv_score:.4f}")
# OOB and CV scores should be close (within ~1%)
```

**Solution 3:**
```python
dt_imp = DecisionTreeClassifier(max_depth=5, random_state=42).fit(X_train, y_train).feature_importances_
rf_imp = RandomForestClassifier(n_estimators=200, random_state=42).fit(X_train, y_train).feature_importances_
gb_imp = GradientBoostingClassifier(n_estimators=200, random_state=42).fit(X_train, y_train).feature_importances_

for name in np.array(data.feature_names)[np.argsort(rf_imp)[-5:]]:
    idx = list(data.feature_names).index(name)
    print(f"{name:30s}  Tree: {dt_imp[idx]:.4f}  RF: {rf_imp[idx]:.4f}  GB: {gb_imp[idx]:.4f}")
# RF and GB usually agree more than single tree (variance reduction)
```

**Solution 4:**
```python
for lr in [0.01, 0.05, 0.1, 0.3, 0.5]:
    gb = GradientBoostingClassifier(
        n_estimators=500, learning_rate=lr, max_depth=3,
        random_state=42
    )
    gb.fit(X_train, y_train)
    # Use staged_predict_proba for learning curves
    test_loss = [log_loss(y_test, p[:, 1])
                 for p in gb.staged_predict_proba(X_test)]
    print(f"LR={lr:.2f}  Best iter={np.argmin(test_loss)+1}  "
          f"Best loss={min(test_loss):.4f}")
# Lower LR converges slower but often achieves lower final loss
```

---

## 9. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Decision Tree | Intuitive, interpretable, but high variance |
| Gini / Entropy | Split criteria that measure impurity |
| Bagging | Reduces variance via parallel training on bootstraps |
| Random Forest | Bagging + feature subsampling — excellent baseline |
| Boosting | Reduces bias via sequential error correction |
| Gradient Boosting | Builds trees on gradient (pseudo-residuals) of loss |
| Learning Rate | Trade-off: low LR + many trees = better generalization |
| XGBoost/LightGBM | Production-grade boosting with regularization |
