# Clustering: Hierarchical, DBSCAN, K-Means, and K-Means++

## 1. Why This Topic Matters

Clustering is the cornerstone of **unsupervised learning** — discovering hidden structure in unlabeled data. It powers customer segmentation, anomaly detection, image compression, document grouping, gene expression analysis, and recommendation engines. Every data scientist must understand clustering because real-world data is often unlabeled, and clustering provides the first step toward understanding it.

---

## 2. K-Means Clustering

### 2.1 Objective Function

K-Means partitions data into K clusters by minimizing the **Within-Cluster Sum of Squares (WCSS)**:

```
J = Σ_{k=1}^{K} Σ_{x_i ∈ C_k} ||x_i - μ_k||²

where:
  K     = number of clusters
  C_k   = set of points assigned to cluster k
  μ_k   = centroid of cluster k (mean of all points in C_k)
  x_i   = data point i
  ||·|| = Euclidean distance (L2 norm)
```

### 2.2 Algorithm (Lloyd's Algorithm)

```
Algorithm: KMeans(data, K, max_iters)
    # Step 1: Initialize K centroids (randomly from data points)
    μ_1, μ_2, ..., μ_K = random_sample(data, K)

    REPEAT until convergence or max_iters:
        # Step 2: ASSIGNMENT — assign each point to nearest centroid
        FOR each x_i:
            C(x_i) = argmin_k ||x_i - μ_k||²

        # Step 3: UPDATE — recompute centroids
        FOR k = 1 to K:
            μ_k = (1/|C_k|) * Σ_{x_i ∈ C_k} x_i

    RETURN cluster assignments, centroids
```

```
Iteration 1:                Iteration 2:                Final:
    A B    C                   A B    C                   A B    C
     \ /                      |\ /   |                   |\     |
      X  ← centroids           |X    ← centroids           | \   |
     /|\                       |/ \   |                   |  \  |
    / | \                      /   \  |                   |   \ |
   ●  ●  ● ●                 ●  ●  ● ●                 ●   ● ● ●

   (chaotic)                  (improving)                 (converged)
```

### 2.3 K-Means++ Initialization

Standard K-Means with random initialization is sensitive to starting positions and often converges to poor local minima. **K-Means++** uses smart initialization:

```
Algorithm: KMeans++(data, K)
    # Step 1: Choose first centroid uniformly at random
    μ_1 = random_choice(data)

    FOR k = 2 to K:
        # Step 2: For each point, compute distance to nearest centroid
        D(x_i) = min_{j < k} ||x_i - μ_j||²

        # Step 3: Choose next centroid with probability ∝ D(x_i)²
        μ_k = sample(data, probability = D(x_i)² / Σ D(x_j)²)

    RETURN μ_1, μ_2, ..., μ_K  → then run standard K-Means
```

**Why it works:** Points far from existing centroids have higher probability of being selected, spreading centroids apart and leading to faster convergence and better solutions.

### 2.4 Choosing K: The Elbow Method

```
WCSS
  |
  |\
  | \
  |  \         ← "Elbow" point
  |   \___
  |        \________
  └────────────────── K
     1  2  3  4  5  6
              ↑
         Optimal K
```

Compute WCSS for K = 1, 2, ..., 10. The "elbow" is where the rate of decrease sharply changes.

**Silhouette Score** (alternative metric):

```
s(i) = (b(i) - a(i)) / max(a(i), b(i))

where:
  a(i) = mean intra-cluster distance (avg dist to points in same cluster)
  b(i) = mean nearest-cluster distance (avg dist to points in nearest other cluster)

s(i) ∈ [-1, 1]:
  s ≈  1 : well-clustered
  s ≈  0 : on cluster boundary
  s ≈ -1 : possibly mis-clustered
```

---

## 3. Hierarchical Clustering

### 3.1 Agglomerative Clustering (Bottom-Up)

Starts with each point as its own cluster and iteratively merges the closest pair.

```
Step 0:        Step 1:          Step 2:          Step 3:         Step 4:
●  ●  ●  ●    ●● ●  ●         ●●●  ●           ●●●●           ●●●●
A  B  C  D    AB C  D         ABC  D           ABCD            ABCD

Dendrogram:
         ┌────┴────┐
    ┌────┴────┐    D
    ┌┴┐       C
    A B

Cut at height h → determines number of clusters
```

### 3.2 Linkage Criteria

| Linkage | Definition | When to Use |
|---------|-----------|-------------|
| **Ward** | Minimizes total within-cluster variance (like K-Means) | Default; produces compact clusters |
| **Complete** | Max distance between any two points across clusters | Tends to produce compact, equal-sized clusters |
| **Average** | Average distance between all point pairs across clusters | Balanced; general purpose |
| **Single** | Min distance between closest points across clusters | Detects elongated/irregular clusters; prone to chaining |

**Ward's Linkage Formula:**

```
Δ(A, B) = (n_A * n_B) / (n_A + n_B) * ||μ_A - μ_B||²

where:
  n_A, n_B = number of points in clusters A and B
  μ_A, μ_B = centroids of clusters A and B
```

### 3.3 Dendrogram Interpretation

The dendrogram is a tree where the y-axis shows the distance (dissimilarity) at which clusters were merged. Cutting the dendrogram at height `h` gives the corresponding cluster assignments.

```
Distance
  3 ─┬─────┬──────────────────────────
     │     │
  2 ─┤  ┌──┤──┐──────────────────────
     │  │  │  │
  1 ─┤  │  │  │──┐───────────────────
     │  │  │  │  │
  0 ─┤  │  │  │  │───────────────────
     A  B  C  D  E

Cut at h=1.5: {A}, {B,C,D}, {E}  → 3 clusters
Cut at h=2.5: {A,B,C,D}, {E}     → 2 clusters
```

---

## 4. DBSCAN (Density-Based Spatial Clustering)

### 4.1 Why DBSCAN?

K-Means and Hierarchical assume convex, globular clusters. DBSCAN can find **arbitrarily shaped clusters** and automatically detects **noise points**.

```
K-Means finds:            DBSCAN finds:
  ○ ○    ● ●               ○○○●●
  ○ ○    ● ●              ○○○○●●●
                            ○○○○○
  (circles only)          (any shape!) + noise ✕
```

### 4.2 Key Parameters and Definitions

```
ε (epsilon) : neighborhood radius
MinPts      : minimum points to form a dense region

Point types:
  Core point:    has ≥ MinPts within ε radius
  Border point:  within ε of a core point, but < MinPts neighbors
  Noise point:   neither core nor border
```

```
        ε-radius
      ╭───────╮
      │  ● ●  │  ← Core point (center): ≥ MinPts neighbors
      │ ● ⊕ ●│  ← Border point: on edge of dense region
      │  ●    │
      ╰───────╯

      ✕           ← Noise point: no dense neighborhood
```

### 4.3 Algorithm

```
Algorithm: DBSCAN(data, ε, MinPts)
    label all points as "unvisited"

    FOR each unvisited point p:
        mark p as "visited"
        neighbors = region_query(p, ε)    # all points within ε

        IF |neighbors| < MinPts:
            label p as NOISE
        ELSE:
            C = new_cluster()
            expand_cluster(p, neighbors, C, ε, MinPts)

    RETURN labels

Function: expand_cluster(p, neighbors, C, ε, MinPts)
    add p to cluster C
    FOR each q in neighbors:
        IF q is "unvisited":
            mark q as "visited"
            q_neighbors = region_query(q, ε)
            IF |q_neighbors| >= MinPts:
                neighbors = neighbors ∪ q_neighbors
        IF q is not yet a member of any cluster:
            add q to cluster C
```

### 4.4 Choosing ε: k-Distance Graph

```
Distance to k-th nearest neighbor
  |
  |         ___________
  |        /
  |       /
  |      /
  |____ /
  └──────────────────── Sorted points
      ↑
  Optimal ε (k = MinPts - 1)
```

Sort distances to the k-th nearest neighbor for all points and look for the "elbow." Use `sklearn.neighbors.NearestNeighbors` to compute.

---

## 5. Python Implementation

### 5.1 K-Means and K-Means++

```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data (drop labels for unsupervised learning)
data = load_iris()
X = data.data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---- K-Means with Elbow Method ----
wcss = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)           # WCSS
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

print("K   WCSS      Silhouette")
for k, w, s in zip(K_range, wcss, silhouette_scores):
    print(f"{k:2d}  {w:8.2f}  {s:.4f}")

# ---- Compare K-Means init strategies ----
km_random = KMeans(n_clusters=3, init='random', n_init=10, random_state=42)
km_plus = KMeans(n_clusters=3, init='k-means++', n_init=10, random_state=42)

km_random.fit(X_scaled)
km_plus.fit(X_scaled)

print(f"\nRandom init WCSS:    {km_random.inertia_:.2f}")
print(f"K-Means++ init WCSS: {km_plus.inertia_:.2f}")
# K-Means++ typically achieves equal or lower WCSS more consistently
```

### 5.2 Hierarchical Clustering

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Agglomerative Clustering with Ward linkage
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_agg = agg.fit_predict(X_scaled)
print(f"Hierarchical Silhouette: {silhouette_score(X_scaled, labels_agg):.4f}")

# Generate Dendrogram
Z = linkage(X_scaled, method='ward')

plt.figure(figsize=(12, 5))
dendrogram(Z, truncate_mode='lastp', p=15, leaf_rotation=90,
           leaf_font_size=10, show_contracted=True)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.ylabel('Distance')
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=150)

# Compare linkage methods
for method in ['ward', 'complete', 'average', 'single']:
    agg = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = agg.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)
    print(f"  {method:10s} Silhouette: {sil:.4f}")
# Ward typically gives highest silhouette for compact clusters
```

### 5.3 DBSCAN

```python
from sklearn.neighbors import NearestNeighbors

# ---- Determine ε using k-distance graph ----
min_pts = 5  # Common heuristic: 2 * dimensions = 2 * 4 = 8, but 5 works well here
neighbors = NearestNeighbors(n_neighbors=min_pts).fit(X_scaled)
distances, _ = neighbors.kneighbors(X_scaled)
k_distances = np.sort(distances[:, -1])

# Find the "elbow" programmatically
# Look for the point of maximum curvature
from numpy import diff
second_derivative = diff(diff(k_distances))
epsilon = k_distances[np.argmax(second_derivative) + 1]
print(f"Suggested epsilon: {epsilon:.3f}")

# ---- DBSCAN Clustering ----
dbscan = DBSCAN(eps=epsilon, min_samples=min_pts)
labels_db = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels_db)) - (1 if -1 in labels_db else 0)
n_noise = (labels_db == -1).sum()
print(f"Clusters found: {n_clusters}")
print(f"Noise points:   {n_noise}")

# Silhouette score (only if >1 cluster and < n-1 noise)
if n_clusters > 1 and n_noise < len(labels_db):
    mask = labels_db != -1
    sil = silhouette_score(X_scaled[mask], labels_db[mask])
    print(f"DBSCAN Silhouette: {sil:.4f}")
```

### 5.4 Comparing All Methods

```python
from sklearn.datasets import make_moons

# Non-globular data where DBSCAN shines
X_moons, _ = make_moons(n_samples=500, noise=0.1, random_state=42)

methods = {
    'K-Means':      KMeans(n_clusters=2, random_state=42),
    'Hierarchical': AgglomerativeClustering(n_clusters=2, linkage='ward'),
    'DBSCAN':       DBSCAN(eps=0.2, min_samples=5)
}

print("Method        Silhouette   Cluster match with moons?")
for name, model in methods.items():
    labels = model.fit_predict(X_moons)
    if len(set(labels)) > 1:
        sil = silhouette_score(X_moons, labels)
        # Check if clustering separates the two half-moons
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(_, labels)
        print(f"  {name:12s}  {sil:.4f}        ARI={ari:.4f}")
    else:
        print(f"  {name:12s}  N/A (single cluster)")

# DBSCAN achieves ARI ≈ 1.0 on moons data (perfect separation)
# K-Means achieves ARI ≈ 0.0 (fails on non-convex shapes)
```

---

## 6. Real-World ML Connections

- **Customer Segmentation:** Retailers use K-Means on purchase behavior (RFM: Recency, Frequency, Monetary value) to create targeted marketing groups.
- **Image Segmentation:** K-Means clusters pixel colors for image compression — reducing a 16M-color image to 16 colors while preserving structure.
- **Anomaly Detection:** DBSCAN labels outliers as noise — used in network intrusion detection and manufacturing defect detection.
- **Genomics:** Hierarchical clustering groups genes with similar expression patterns, revealing biological pathways.
- **Geospatial Analysis:** DBSCAN clusters GPS coordinates to identify hotspots in ride-sharing (Uber), crime mapping, and earthquake epicenter analysis.

---

## 7. Comparison Table

| Aspect | K-Means | Hierarchical | DBSCAN |
|--------|---------|-------------|--------|
| Shape assumption | Spherical | Any (depends on linkage) | Any (density-based) |
| Need to specify K | Yes | No (but need cut height) | No |
| Handles noise | No | No | Yes (labels as -1) |
| Complexity | O(nKd) per iter | O(n²) or O(n² log n) | O(n log n) with index |
| Scalability | Excellent | Poor for large n | Good with spatial index |
| Deterministic | No (random init) | Yes | Yes |
| Cluster sizes | Similar | Flexible | Variable |

---

## 8. Exercises

### Exercise 1: Optimal K Selection
Using the iris dataset (scaled), compute WCSS and silhouette scores for K=1 to 10. Plot both. Which K is optimal by each metric?

### Exercise 2: DBSCAN Sensitivity
Test DBSCAN on make_moons with eps values [0.05, 0.1, 0.2, 0.3, 0.5] and min_samples values [3, 5, 10, 20]. Report number of clusters and noise points for each combination.

### Exercise 3: High-Dimensional Clustering
Load sklearn's digits dataset (64 features, 10 classes). Apply K-Means with K=10. Compare silhouette score before and after PCA to 10 dimensions. Does dimensionality reduction help?

### Solutions

**Solution 1:**
```python
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_iris().data)
results = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X)
    sil = silhouette_score(X, km.labels_)
    results.append((k, km.inertia_, sil))
    print(f"K={k:2d}  WCSS={km.inertia_:8.2f}  Silhouette={sil:.4f}")

# Elbow typically at K=2 or K=3; silhouette peak usually at K=2
# For iris (3 species), K=3 is the "correct" answer
```

**Solution 2:**
```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN

X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
print("eps   min_samples  clusters  noise")
for eps in [0.05, 0.1, 0.2, 0.3, 0.5]:
    for ms in [3, 5, 10, 20]:
        db = DBSCAN(eps=eps, min_samples=ms).fit(X)
        n_cl = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
        n_noise = (db.labels_ == -1).sum()
        print(f"{eps:.2f}  {ms:12d}  {n_cl:9d}  {n_noise:5d}")
# eps=0.2, min_samples=5 usually gives 2 clusters, ~0 noise
```

**Solution 3:**
```python
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA

X = load_digits().data
X_scaled = StandardScaler().fit_transform(X)

# Without PCA
km = KMeans(n_clusters=10, random_state=42)
labels_no_pca = km.fit_predict(X_scaled)
sil_no_pca = silhouette_score(X_scaled, labels_no_pca)
print(f"Without PCA Silhouette: {sil_no_pca:.4f}")

# With PCA
X_pca = PCA(n_components=10).fit_transform(X_scaled)
labels_pca = KMeans(n_clusters=10, random_state=42).fit_predict(X_pca)
sil_pca = silhouette_score(X_pca, labels_pca)
print(f"With PCA (10d) Silhouette: {sil_pca:.4f}")
# PCA typically improves silhouette by removing noise dimensions
```

---

## 9. Summary

| Method | When to Use |
|--------|------------|
| K-Means | Large datasets, spherical clusters, when K is known/approximate |
| K-Means++ | Always use instead of random init for better, more consistent results |
| Hierarchical | Small datasets, when you want a tree of nested clusters |
| DBSCAN | Arbitrarily shaped clusters, when noise detection is needed, spatial data |
| Silhouette Score | General metric for cluster quality (-1 to 1, higher is better) |
| Elbow Method | Quick heuristic for choosing K in K-Means |
