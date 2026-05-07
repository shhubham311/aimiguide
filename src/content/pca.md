# Dimensionality Reduction & Principal Component Analysis (PCA)

## 1. Why This Topic Matters

Real-world datasets often have hundreds or thousands of features (genes, pixels, sensor readings), leading to the **curse of dimensionality**: distances become meaningless, models overfit, and computation explodes. Dimensionality reduction is the art of finding the most informative low-dimensional representation of data. PCA is the most fundamental technique — it underpins facial recognition (Eigenfaces), gene expression analysis, data compression, and noise filtering. Understanding PCA is also essential before tackling advanced methods like t-SNE, UMAP, and autoencoders.

---

## 2. The Curse of Dimensionality

As the number of dimensions `d` increases:

- **Volume grows exponentially:** The volume of a unit hypercube is `1^d = 1`, but the fraction near any surface grows.
- **Distances concentrate:** All points become roughly equidistant.
- **Data becomes sparse:** To maintain density, you need exponentially more samples (`N ∝ 2^d`).

```
       1D: ●●●●●●●●●●    Dense
       2D: ● ● ●          Sparse
           ● ● ●
           ● ● ●
       3D: Even sparser
      100D: Nearly empty!
```

**Rule of thumb:** You need at least `10*d` samples to reliably estimate a model in `d` dimensions.

---

## 3. Intuition Behind PCA

PCA finds the directions (axes) along which the data varies the most, then projects the data onto these axes.

```
Original 2D Data:              After PCA (1D):
     y                             |
     |   ●  ●  ●                  |   ●
     |  ●   ●   ●                 |  ● ●
     | ●    PC1→    ●             | ●   ●  → PC1 captures max variance
     |      ●  ●                  |  ● ●
     |   ●                        |   ●
     └────────── x                └────────── x

PC1 (1st Principal Component): direction of maximum variance
PC2 (2nd Principal Component): orthogonal to PC1, next max variance
```

**Key Properties:**
- Principal components are **orthogonal** (uncorrelated).
- They are ordered by **variance explained** (PC1 > PC2 > ... > PCd).
- PCA is a **linear** transformation — it cannot capture curved manifolds.

---

## 4. Mathematical Foundation

### 4.1 Step-by-Step PCA Derivation

Given a dataset `X` with `n` samples and `d` features:

**Step 1: Center the data** (subtract mean of each feature)

```
X_centered = X - X_bar

where X_bar = (1/n) * Σ X_i   (row vector of feature means)
```

**Step 2: Compute the covariance matrix**

```
Σ = (1/(n-1)) * X_centered^T * X_centered

Σ is a d × d symmetric matrix where:
  Σ[i,j] = covariance between feature i and feature j
  Σ[i,i] = variance of feature i
```

**Step 3: Eigendecomposition**

Find eigenvalues and eigenvectors of Σ:

```
Σ * v_k = λ_k * v_k

where:
  v_k = k-th eigenvector (principal direction)
  λ_k = k-th eigenvalue (variance along v_k)
```

**Step 4: Select top K components**

Sort eigenvectors by eigenvalue (descending). Keep the top K:

```
W = [v_1, v_2, ..., v_K]    (d × K matrix of top eigenvectors)
```

**Step 5: Project data**

```
Z = X_centered * W           (n × K projected data)

Z[i, k] = the k-th principal component score for sample i
```

### 4.2 Variance Explained

```
Total variance     = Σ λ_k = trace(Σ)
Variance of PC_k   = λ_k
% explained by PC_k = λ_k / Σ λ_j

Cumulative variance explained by top K PCs:
  R²_K = Σ_{k=1}^{K} λ_k / Σ_{j=1}^{d} λ_j
```

```
Cumulative Variance Explained
  1.0 ┤████████████████████████
      │████████████████████████████
  0.8 ┤████████████████████████████
      │████████████████████████████████
  0.6 ┤████████████████████████████████
      │
  0.4 ┤
      │
  0.2 ┤
      │
  0.0 ┼──┬──┬──┬──┬──┬──┬──┬──┬──┬──
      PC1 PC2 PC3 PC4 PC5 PC6 PC7 PC8 PC9 PC10

  "Elbow" determines K needed for 95% variance
```

### 4.3 Singular Value Decomposition (SVD) Formulation

In practice, sklearn uses SVD instead of eigendecomposition (more numerically stable):

```
X_centered = U * S * V^T

where:
  U  = n × n orthogonal matrix (left singular vectors)
  S  = n × d diagonal matrix of singular values
  V^T = d × d orthogonal matrix (right singular vectors = principal directions)

Relationship to eigenvalues:
  λ_k = s_k² / (n - 1)

PCA projection:
  Z = X_centered * V_K    where V_K contains first K columns of V
```

---

## 5. PCA Properties and Limitations

### 5.1 What PCA Preserves

| Property | Preserved? |
|----------|-----------|
| Euclidean distances (approximate) | Partially |
| Pairwise angles | No |
| Global structure | Yes (linear) |
| Local neighborhood | Yes (mostly) |
| Cluster separation | Can improve or degrade |

### 5.2 Limitations

- **Linear only:** Cannot capture non-linear manifolds (use Kernel PCA, t-SNE, or UMAP).
- **Interpretability:** Principal components are linear combinations of all original features — hard to interpret.
- **Variance ≠ Importance:** High variance may not correlate with predictive importance.
- **Scale sensitivity:** Must standardize features before PCA (unless all features share the same scale).

```
Standardization requirement:

Feature A: range [0, 1]        Feature B: range [0, 100000]
     ●●●●●                         ●●●●●

Without standardization: PCA will capture Feature B (numerically larger)
With standardization: PCA captures the true underlying structure
```

---

## 6. Python Implementation

### 6.1 Basic PCA on Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Load and standardize
data = load_iris()
X = data.data
y = data.target
X_scaled = StandardScaler().fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Explained Variance Ratio:")
for i, evr in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {evr:.4f} ({evr*100:.2f}%)")
print(f"Total: {sum(pca.explained_variance_ratio_):.4f}")
# Typically: PC1≈72.8%, PC2≈22.9%, Total≈95.8%

print(f"\nCumulative: {np.cumsum(pca.explained_variance_ratio_)}")
print(f"\nPrincipal Components (loadings):")
print(f"  PC1 = {dict(zip(data.feature_names, np.round(pca.components_[0], 3)))}")
print(f"  PC2 = {dict(zip(data.feature_names, np.round(pca.components_[1], 3)))}")
```

### 6.2 Choosing Number of Components

```python
# Full PCA to see all components
pca_full = PCA().fit(X_scaled)

cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
n_95 = np.argmax(cumulative_var >= 0.95) + 1
n_99 = np.argmax(cumulative_var >= 0.99) + 1

print("Component  Variance_Ratio  Cumulative")
for i, (vr, cv) in enumerate(zip(
        pca_full.explained_variance_ratio_, cumulative_var)):
    print(f"  PC{i+1:2d}       {vr:.4f}          {cv:.4f}")

print(f"\nComponents for 95% variance: {n_95}")
print(f"Components for 99% variance: {n_99}")
print(f"Components for 100% variance: {X_scaled.shape[1]}")

# Scree plot
eigenvalues = pca_full.explained_variance_
print(f"\nEigenvalues: {np.round(eigenvalues, 3)}")
# Kaiser criterion: keep components with eigenvalue > 1
kaiser = np.sum(eigenvalues > 1)
print(f"Kaiser criterion: keep {kaiser} components")
```

### 6.3 PCA for Data Compression (MNIST-like)

```python
from sklearn.datasets import load_digits

digits = load_digits()
X = digits.data  # 1797 samples × 64 features (8×8 images)
X_scaled = StandardScaler().fit_transform(X)

# Compare compression levels
for n_comp in [5, 10, 20, 32, 64]:
    pca = PCA(n_components=n_comp)
    X_reduced = pca.fit_transform(X_scaled)
    X_reconstructed = pca.inverse_transform(X_reduced)

    # Reconstruction error
    mse = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"  Components: {n_comp:2d}  |  Variance: {sum(pca.explained_variance_ratio_)*100:.1f}%  |  "
          f"Reconstruction MSE: {mse:.4f}  |  Compression: {64/n_comp:.1f}x")

# Visualize reconstruction for one sample
pca_10 = PCA(n_components=10)
X_red = pca_10.fit_transform(X_scaled)
X_rec = pca_10.inverse_transform(X_red)

sample_idx = 0
print(f"\nOriginal pixel values (sample 0, first 8):")
print(f"  {np.round(X_scaled[sample_idx, :8], 2)}")
print(f"Reconstructed (10 PCs, first 8):")
print(f"  {np.round(X_rec[sample_idx, :8], 2)}")
```

### 6.4 PCA Before Classification

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

digits = load_digits()
X, y = digits.data, digits.target

# Without PCA
rf = RandomForestClassifier(n_estimators=100, random_state=42)
cv_no_pca = cross_val_score(rf, X, y, cv=5)
print(f"RF (64 features):   Mean={cv_no_pca.mean():.4f}  Std={cv_no_pca.std():.4f}")

# With PCA pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=20)),
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
])
cv_pca = cross_val_score(pipe, X, y, cv=5)
print(f"RF (20 PCA features): Mean={cv_pca.mean():.4f}  Std={cv_pca.std():.4f}")

# Speed comparison
import time
t0 = time.time()
rf.fit(X, y)
t1 = time.time()
pipe.fit(X, y)
t2 = time.time()
print(f"\nTraining time without PCA: {t1-t0:.3f}s")
print(f"Training time with PCA:    {t2-t1:.3f}s")
```

### 6.5 Kernel PCA (Non-linear Dimensionality Reduction)

```python
from sklearn.decomposition import KernelPCA
from sklearn.datasets import make_circles

# Non-linear data: concentric circles
X_circles, y_circles = make_circles(n_samples=500, noise=0.05, factor=0.5, random_state=42)

# Linear PCA fails
pca_linear = PCA(n_components=2)
X_pca_linear = pca_linear.fit_transform(X_circles)

# Kernel PCA succeeds (RBF kernel)
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_kpca = kpca.fit_transform(X_circles)

from sklearn.metrics import silhouette_score
sil_linear = silhouette_score(X_circles, pca_linear.transform(X_circles)[:, 0].reshape(-1,1))
# Kernel PCA can separate circles when linear PCA cannot
print(f"Linear PCA 1D separation: difficult to distinguish circles")
print(f"Kernel PCA: can linearly separate the circles in kernel space")
print(f"KPCA component 1 range: [{X_kpca[:,0].min():.2f}, {X_kpca[:,0].max():.2f}]")
print(f"KPCA component 2 range: [{X_kpca[:,1].min():.2f}, {X_kpca[:,1].max():.2f}]")
```

---

## 7. Real-World ML Connections

- **Facial Recognition (Eigenfaces):** PCA on face images extracts "eigenfaces" — the principal directions of facial variation. Each face is a combination of eigenfaces.
- **Genomics:** Gene expression datasets have 20,000+ genes (features) but only ~100 samples. PCA reduces to 10–50 components before clustering or classification.
- **Finance:** PCA on stock returns extracts "market factors" — the first PC often represents the overall market movement.
- **Image Compression:** JPEG compression uses a related technique (DCT, not PCA, but same principle) to represent images with fewer coefficients.
- **Data Visualization:** PCA to 2D or 3D is the first step for visualizing high-dimensional data in scatter plots.

---

## 8. Advanced: Incremental PCA and Sparse PCA

### 8.1 Incremental PCA

For datasets too large for memory:

```python
from sklearn.decomposition import IncrementalPCA

# Process data in batches
ipca = IncrementalPCA(n_components=20, batch_size=200)
X_ipca = ipca.fit(X_scaled)

print(f"Variance explained: {sum(ipca.explained_variance_ratio_)*100:.1f}%")
# Results match standard PCA for same n_components
```

### 8.2 Sparse PCA

Standard PCA components load on ALL features. Sparse PCA forces loadings to be zero for most features, improving interpretability:

```python
from sklearn.decomposition import SparsePCA

spca = SparsePCA(n_components=5, alpha=1.0, random_state=42)
X_spca = spca.fit_transform(X_scaled)

# Count non-zero loadings per component
for i, comp in enumerate(spca.components_):
    n_nonzero = np.sum(np.abs(comp) > 0.01)
    print(f"Component {i+1}: {n_nonzero} non-zero loadings "
          f"(of {X_scaled.shape[1]} features)")
# Sparse PCA: typically 5-15 non-zero loadings vs 64 for regular PCA
```

---

## 9. Exercises

### Exercise 1: Dimensionality Reduction Sweep
Using the digits dataset, apply PCA with n_components from 2 to 50 (step 2). Plot cumulative variance explained. Report the minimum components needed for 90%, 95%, and 99% variance.

### Exercise 2: PCA + KNN Classification
Use KNN classifier (K=5) on the digits dataset. Compare accuracy with original 64 features vs PCA-reduced features at 10, 20, 30, and 40 components. Which gives the best accuracy-to-dimensionality trade-off?

### Exercise 3: Noise Filtering with PCA
Load the digits dataset, add Gaussian noise (np.random.normal(0, 4, X.shape)), then use PCA to filter the noise: keep enough components for 50% variance, inverse_transform, and compute reconstruction MSE against the original clean data.

### Exercise 4: Feature Correlation Analysis
Compute the covariance matrix of the iris dataset. Apply PCA and verify that the principal components are uncorrelated by computing the covariance of the PCA-transformed data.

### Solutions

**Solution 1:**
```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

X = StandardScaler().fit_transform(load_digits().data)
pca = PCA().fit(X)
cumvar = np.cumsum(pca.explained_variance_ratio_)

for threshold in [0.90, 0.95, 0.99]:
    n = np.argmax(cumvar >= threshold) + 1
    print(f"{threshold*100:.0f}% variance: {n} components")
# Typical results: 90%≈21, 95%≈30, 99%≈43
```

**Solution 2:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

X, y = load_digits(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=5)
cv_orig = cross_val_score(knn, X, y, cv=5)
print(f"Original (64d): {cv_orig.mean():.4f}")

for nc in [10, 20, 30, 40]:
    pipe = Pipeline([('scaler', StandardScaler()),
                     ('pca', PCA(n_components=nc)),
                     ('knn', KNeighborsClassifier(n_neighbors=5))])
    cv = cross_val_score(pipe, X, y, cv=5)
    print(f"PCA ({nc:2d}d):     {cv.mean():.4f}")
# Typically: 10d≈0.96, 20d≈0.97, 30d≈0.97, original≈0.97
# Best trade-off: 20 components (97% accuracy at 31% of original dims)
```

**Solution 3:**
```python
X_clean = load_digits().data
X_clean_scaled = StandardScaler().fit_transform(X_clean)

noise = np.random.normal(0, 4, X_clean.shape)
X_noisy = X_clean_scaled + noise
X_noisy_pca = StandardScaler().fit_transform(X_noisy)

# Keep 50% variance
pca = PCA(n_components=0.50, svd_solver='full')
X_filtered = pca.inverse_transform(pca.fit_transform(X_noisy_pca))

mse_noisy = np.mean((X_noisy_pca - X_clean_scaled)**2)
mse_filtered = np.mean((X_filtered - X_clean_scaled)**2)
print(f"MSE (noisy):  {mse_noisy:.4f}")
print(f"MSE (filtered): {mse_filtered:.4f}")
print(f"Improvement:   {(1 - mse_filtered/mse_noisy)*100:.1f}%")
```

**Solution 4:**
```python
from sklearn.datasets import load_iris

X = StandardScaler().fit_transform(load_iris().data)

# Covariance of original data
cov_orig = np.cov(X, rowvar=False)
print(f"Original cov matrix max off-diagonal: {np.max(np.abs(cov_orig - np.diag(np.diag(cov_orig)))):.4f}")

# PCA transform
X_pca = PCA().fit_transform(X)
cov_pca = np.cov(X_pca, rowvar=False)
print(f"PCA cov matrix max off-diagonal:     {np.max(np.abs(cov_pca - np.diag(np.diag(cov_pca)))):.2e}")
# Off-diagonal elements should be ~0 (up to floating point precision)
```

---

## 10. Summary

| Concept | Key Takeaway |
|---------|-------------|
| Curse of Dimensionality | High dimensions make distances meaningless and models overfit |
| PCA | Linear projection that maximizes variance along orthogonal axes |
| Eigenvalues = Variance | Each PC's eigenvalue equals the variance captured by that component |
| Explained Variance | Choose K components that capture 95–99% of total variance |
| Standardization | Always standardize before PCA when features have different scales |
| SVD | Numerically stable alternative to eigendecomposition; used by sklearn |
| Kernel PCA | Extends PCA to non-linear patterns using the kernel trick |
| Noise Filtering | PCA can denoise data by discarding low-variance components |
| Compression | Represent data with fewer numbers while preserving structure |
