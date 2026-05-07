# Introduction: Supervised & Unsupervised Learning

## Why This Topic Matters

Machine Learning (ML) is the engine behind virtually every modern AI system. Whether you are building a spam filter, a self-driving car, or a medical diagnosis tool, the first and most critical decision you make is: **what type of learning problem am I solving?** Getting this wrong means choosing the wrong algorithm, the wrong evaluation strategy, and ultimately building a system that fails in production. Understanding the fundamental taxonomy of ML—supervised, unsupervised, and the boundary between them—is not optional; it is the foundation upon which all subsequent techniques are built.

In industry, misclassifying a problem type has real consequences. A team that treats a clustering problem as a classification problem will waste months labeling data that doesn't need labels. Conversely, a team that tries to use unsupervised learning when precise predictions are needed will deliver a model that cannot meet business requirements. This lesson gives you the mental models to avoid these costly mistakes.

---

## The Big Picture: Types of Machine Learning

```
                        Machine Learning
                             |
              +--------------+--------------+
              |                             |
         Supervised                   Unsupervised
         Learning                    Learning
              |                             |
    +---------+---------+         +--------+--------+
    |                   |         |                 |
Classification     Regression   Clustering    Dim. Reduction
    |                   |         |                 |
(Spam/Not Spam)   (House Prices) (Customer Segs)  (PCA)
```

There is also **Reinforcement Learning** (an agent learns by interacting with an environment) and **Semi-Supervised Learning** (a mix of labeled and unlabeled data), but supervised and unsupervised learning cover the vast majority of real-world ML tasks.

---

## Supervised Learning

### Definition

In **supervised learning**, the algorithm learns from **labeled data**. Each training example consists of an input (features) and a known output (target/label). The goal is to learn a mapping function `f` such that:

```
f: X → y
```

where `X` is the feature matrix and `y` is the target vector. Given new, unseen data, the model applies `f` to predict the output.

### The Two Flavors: Classification vs. Regression

| Aspect | Classification | Regression |
|--------|---------------|------------|
| Target type | Discrete categories | Continuous values |
| Example | Spam / Not Spam | House price in dollars |
| Output | Class label | Real number |
| Typical metrics | Accuracy, F1, AUC | MSE, R², MAE |

### How Supervised Learning Works (Step by Step)

```
Step 1: Collect labeled data
    [ (x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ) ]

Step 2: Split into training and testing sets
    Training: 80%    Testing: 20%

Step 3: Choose a model (e.g., Linear Regression, Decision Tree)

Step 4: Train the model on training data
    Model learns the mapping f(X) → y

Step 5: Evaluate on test data
    Compare predictions ŷ vs. true labels y

Step 6: Deploy and monitor
```

### Visual: Supervised Learning Pipeline

```
  Raw Data          Training Set         Model          Predictions
 ┌─────────┐     ┌──────────────┐    ┌──────────┐    ┌──────────┐
 │ x₁ → y₁ │     │  x_train     │    │          │    │  x_new   │
 │ x₂ → y₂ │ ──▶ │  y_train     │ ──▶│  f(x)    │──▶ │  ŷ_pred  │
 │ x₃ → y₃ │     │              │    │          │    │          │
 │  ...    │     └──────────────┘    └──────────┘    └──────────┘
 └─────────┘
```

### Python Example: Supervised Classification with Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Load labeled data
iris = load_iris()
X = iris.data       # Features: sepal length, sepal width, petal length, petal width
y = iris.target     # Labels: 0 (setosa), 1 (versicolor), 2 (virginica)

print(f"Feature shape: {X.shape}")   # (150, 4)
print(f"Target shape: {y.shape}")    # (150,)
print(f"Classes: {iris.target_names}")

# Step 2: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 3: Choose and train a model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predict and evaluate
y_pred = model.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# Step 5: Predict on new, unseen data
new_flower = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(new_flower)
print(f"\nPrediction for {new_flower}: {iris.target_names[prediction[0]]}")
```

**Output:**
```
Feature shape: (150, 4)
Target shape: (150,)
Classes: ['setosa' 'versicolor' 'virginica']

Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        10
   virginica       1.00      1.00      1.00        10

    accuracy                           1.00        30

Prediction for [[5.1, 3.5, 1.4, 0.2]]: setosa
```

---

## Unsupervised Learning

### Definition

In **unsupervised learning**, the algorithm works with **unlabeled data**. There are no target values `y`. The goal is to discover hidden patterns, structures, or relationships in the data.

```
Supervised:    f: (X, y) → learned mapping        (labels known)
Unsupervised:  f: X → structure / patterns        (no labels)
```

### Key Unsupervised Tasks

1. **Clustering**: Grouping similar data points together.
2. **Dimensionality Reduction**: Compressing features while preserving information.
3. **Anomaly Detection**: Finding unusual data points.
4. **Association Rules**: Discovering rules (e.g., "customers who bought X also bought Y").

### Clustering in Depth: K-Means Algorithm

K-Means is the most widely used clustering algorithm. It partitions data into `k` clusters by iteratively:

1. **Initializing** `k` centroids (randomly or via k-means++).
2. **Assigning** each point to the nearest centroid.
3. **Updating** centroids to the mean of assigned points.
4. **Repeating** steps 2-3 until convergence.

#### Mathematical Formulation

Given `n` data points `{x₁, x₂, ..., xₙ}`, find `k` cluster centers `{μ₁, μ₂, ..., μₖ}` that minimize the **Within-Cluster Sum of Squares (WCSS)**:

```
        k     n
WCSS = Σ   Σ   ||xᵢ - μⱼ||²
       j=1  i: xᵢ ∈ Cⱼ
```

where:
- `k` = number of clusters (hyperparameter chosen by the user)
- `Cⱼ` = set of points assigned to cluster `j`
- `μⱼ` = centroid of cluster `j` (mean of all points in `Cⱼ`)
- `||xᵢ - μⱼ||²` = squared Euclidean distance between point `xᵢ` and centroid `μⱼ`

#### Visual: K-Means Iteration

```
Iteration 1              Iteration 2              Iteration 3 (Converged)
    ●                         ●                        ●
  ●   ●  ▲                  ●   ●  ▲                 ●   ●  ▲
    ▲      ●    →              ▲      ●    →             ▲      ●
  ●     ▲                    ●     ▲                   ●     ▲
      ●    ●                    ●    ●                     ●    ●

  Random centroids          Points reassigned        Stable clusters
  (× marks)                 to nearest centroid      found
```

### Python Example: K-Means Clustering on Customer Data

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# Generate synthetic customer segmentation data (300 customers, 4 features)
X, _ = make_blobs(
    n_samples=300, centers=4, cluster_std=1.2,
    random_state=42, n_features=4
)

# Feature names for context
feature_names = ['Annual Income', 'Monthly Spend', 'Website Visits', 'Support Tickets']

# Step 1: Always scale your data for distance-based algorithms
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Find optimal k using the Elbow Method
wcss = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

# Step 3: Choose k=4 (we know it's 4 from generation, but in practice use elbow)
optimal_k = 4
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

print(f"Cluster centers (scaled):\n{kmeans.cluster_centers_.round(2)}")
print(f"\nCluster sizes: {np.bincount(kmeans.labels_)}")
print(f"Silhouette Score: {silhouette_score(X_scaled, kmeans.labels_):.4f}")
print(f"WCSS (Inertia): {kmeans.inertia_:.2f}")

# Step 4: Assign new customers to clusters
new_customers = np.array([[50000, 200, 15, 2], [20000, 50, 3, 8]])
new_scaled = scaler.transform(new_customers)
clusters = kmeans.predict(new_scaled)
print(f"\nNew customer cluster assignments: {clusters}")
```

**Output:**
```
Cluster centers (scaled):
[[-1.12 -1.07 -1.13 -1.06]
 [ 1.13  1.12  1.16  1.08]
 [-0.07  0.11 -0.01  0.03]
 [ 1.51 -0.91  1.27 -1.  ]]

Cluster sizes: [75 75 75 75]
Silhouette Score: 0.5672
WCSS (Inertia): 724.84

New customer cluster assignments: [1 0]
```

---

## Dimensionality Reduction: PCA

**Principal Component Analysis (PCA)** reduces the number of features while preserving maximum variance. It finds new axes (principal components) that capture the most information.

Mathematically, PCA solves:

```
Find orthonormal vectors v₁, v₂, ..., v_d such that:
    v₁ maximizes Var(Xv₁)
    v₂ maximizes Var(Xv₂) subject to v₂ ⊥ v₁
    ...

This is equivalent to finding the top d eigenvectors of the covariance matrix:
    Σ = (1/n) X^T X
```

```python
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

# Load handwritten digits (64 features = 8x8 pixel images)
digits = load_digits()
X = digits.data
print(f"Original shape: {X.shape}")  # (1797, 64)

# Reduce to 2 dimensions for visualization
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print(f"Reduced shape: {X_reduced.shape}")  # (1797, 2)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")

# How many components to explain 95% variance?
pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X)
print(f"\nComponents for 95% variance: {pca_95.n_components_}")
```

**Output:**
```
Original shape: (1797, 64)
Reduced shape: (1797, 2)
Explained variance ratio: [0.1489 0.1362]
Total variance explained: 0.2851

Components for 95% variance: 29
```

---

## Supervised vs. Unsupervised: Side-by-Side Comparison

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│      Aspect         │    Supervised        │    Unsupervised      │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Data Labels         │ Required (expensive) │ Not needed           │
│ Goal                │ Predict outcomes     │ Find patterns        │
│ Complexity          │ Often simpler        │ More exploratory     │
│ Evaluation          │ Clear metrics        │ Subjective/indirect  │
│ Examples            │ Classification,      │ Clustering, PCA,     │
│                     │ Regression           │ Anomaly Detection    │
│ Data needed         │ Less (labeled)       │ More (unlabeled OK)  │
│ Human effort        │ High (labeling)      │ Low                  │
└─────────────────────┴──────────────────────┴──────────────────────┘
```

---

## Real-World Applications

### Supervised Learning in Production
- **Medical Imaging**: Classifying X-rays as normal or abnormal (Google Health).
- **Fraud Detection**: Predicting whether a transaction is fraudulent (Visa, Stripe).
- **Recommendation Systems**: Predicting user ratings (Netflix, Spotify).
- **Natural Language Processing**: Sentiment analysis, named entity recognition.

### Unsupervised Learning in Production
- **Customer Segmentation**: Grouping users for targeted marketing (Amazon, Spotify).
- **Topic Modeling**: Discovering themes in documents (Google News).
- **Anomaly Detection**: Identifying network intrusions or manufacturing defects.
- **Feature Learning**: Autoencoders pre-training features for downstream tasks.

---

## The Bias-Variance Tradeoff (Preview)

Every ML model must balance two sources of error:

```
Total Error = Bias² + Variance + Irreducible Noise

High Bias (Underfitting):  Model is too simple, misses patterns
High Variance (Overfitting): Model is too complex, memorizes noise

    Error
      │
      │    Variance ──────╱
      │                  ╱
      │   Bias²    ────╱
      │               ╱
      │              ╱──── Irreducible
      │             ╱
      └──────────────────────── Model Complexity
              Optimal
```

This tradeoff is central to everything we will study. It applies equally to supervised and unsupervised methods (e.g., choosing `k` in K-Means).

---

## Exercises

### Exercise 1: Identify the Learning Type

For each scenario, classify it as **supervised** (classification or regression) or **unsupervised** (clustering, dimensionality reduction, or anomaly detection):

1. Predicting the price of a house based on square footage, bedrooms, and location.
2. Grouping news articles into topics without prior labels.
3. Detecting fraudulent credit card transactions.
4. Identifying unusual patterns in server logs.
5. Recommending products based on user purchase history.

### Solution 1:

1. **Supervised → Regression**: The target (price) is continuous.
2. **Unsupervised → Clustering**: No labels; finding groups.
3. **Supervised → Classification**: Labels (fraud/not fraud) exist; binary outcome.
4. **Unsupervised → Anomaly Detection**: Finding outliers without labeled anomalies.
5. **Supervised → Classification/Regression**: Uses historical purchase data with known outcomes.

### Exercise 2: Implement and Compare K-Means with Different `k`

```python
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Generate data with 3 true clusters
X, true_labels = make_blobs(n_samples=500, centers=3, cluster_std=1.5, random_state=42)

# TODO: Run KMeans for k=2,3,4,5,6
# TODO: Plot WCSS (elbow) and silhouette scores
# TODO: Which k is optimal?

results = []
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    sil = silhouette_score(X, km.labels_)
    results.append({'k': k, 'wcss': km.inertia_, 'silhouette': sil})
    print(f"k={k}: WCSS={km.inertia_:.2f}, Silhouette={sil:.4f}")
```

### Solution 2:

```
k=2: WCSS=1652.31, Silhouette=0.5214
k=3: WCSS=925.48,  Silhouette=0.5921   ← Optimal (matches true clusters)
k=4: WCSS=783.15,  Silhouette=0.4987
k=5: WCSS=670.23,  Silhouette=0.4532
k=6: WCSS=587.91,  Silhouette=0.4108
```

The elbow occurs at `k=3` (WCSS drops sharply then plateaus), and the silhouette score peaks at `k=3`, confirming the optimal number of clusters.

### Exercise 3: Build a Supervised Model and Evaluate

Using the breast cancer dataset (`sklearn.datasets.load_breast_cancer`), build a classifier that predicts malignant vs. benign tumors. Split the data, train a model, and report accuracy.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
# Expected: ~0.965
```

---

## Key Takeaways

1. **Supervised learning** requires labeled data and is used for prediction (classification and regression).
2. **Unsupervised learning** discovers hidden structure in unlabeled data (clustering, dimensionality reduction).
3. The choice between supervised and unsupervised depends on whether you have labels and what your goal is.
4. **Always split data** into training and test sets for supervised learning.
5. **Always scale features** before distance-based algorithms (K-Means, KNN).
6. The **bias-variance tradeoff** governs model complexity decisions across all ML methods.

---

*Next: [Linear Regression & Ridge Regression](./linear-regression.md)*
