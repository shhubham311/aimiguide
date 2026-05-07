# Data Visualization using Matplotlib & Seaborn

## Table of Contents
1. [Introduction to Data Visualization](#introduction-to-data-visualization)
2. [Matplotlib Fundamentals](#matplotlib-fundamentals)
3. [Seaborn for Statistical Graphics](#seaborn-for-statistical-graphics)
4. [Distribution Plots](#distribution-plots)
5. [Relationship Plots](#relationship-plots)
6. [Categorical Plots](#categorical-plots)
7. [Time Series Visualization](#time-series-visualization)
8. [Heatmaps & Correlation](#heatmaps--correlation)
9. [Advanced Customization](#advanced-customization)
10. [Dashboard & Multi-Panel Layouts](#dashboard--multi-panel-layouts)
11. [Interactive Visualization](#interactive-visualization)
12. [Visualization Best Practices](#visualization-best-practices)
13. [Exercises with Solutions](#exercises-with-solutions)
14. [ML Connections](#ml-connections)

---

## Introduction to Data Visualization

Data visualization is the graphical representation of data and information. It is a critical skill for data scientists to explore data, communicate findings, and validate models.

### Why Visualize Data?

```
┌──────────────────────────────────────────────────────────┐
│              The Visualization Pipeline                   │
│                                                            │
│   Raw Data ──> Clean Data ──> Explore ──> Insight ──> Act │
│                   │              │                         │
│                   │         ┌────┴────┐                    │
│                   │         │Visualize│                    │
│                   │         └────┬────┘                    │
│                   │              │                         │
│                   └──────────────┘                         │
│                                                            │
│   Goals:                                                   │
│   ├── Detect patterns, trends, outliers                    │
│   ├── Understand distributions                            │
│   ├── Compare groups                                       │
│   ├── Show relationships between variables                 │
│   └── Communicate results to stakeholders                  │
└──────────────────────────────────────────────────────────┘
```

### Anscombe's Quartet: Why Visualization Matters

```python
import numpy as np

# Four datasets with identical statistics but very different patterns
anscombe = {
    'I':    {'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
             'y': [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]},
    'II':   {'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
             'y': [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]},
    'III':  {'x': [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5],
             'y': [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]},
    'IV':   {'x': [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
             'y': [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]}
}

# All four have: mean x = 9, mean y ≈ 7.50
# std x = 3.32, std y ≈ 2.03
# correlation ≈ 0.816, regression line: y ≈ 3.00 + 0.50x
```

### Matplotlib vs Seaborn

| Feature              | Matplotlib              | Seaborn                     |
|---------------------|------------------------|-----------------------------|
| Level               | Low-level              | High-level (built on mpl)   |
| Syntax              | Verbose                | Concise                     |
| Defaults            | Basic                  | Statistically-aware         |
| Best for            | Full customization     | Statistical exploration     |
| Integration         | Base for most libs     | Works with Pandas natively  |
| Learning curve      | Moderate               | Easy                        |

```
       Matplotlib (Foundation)
              │
       ┌──────┼──────┐
       │      │      │
    Seaborn  Plotly  Bokeh
    (stats) (inter.) (web)
```

---

## Matplotlib Fundamentals

### Figure and Axes Architecture

```
┌──────────────────────────────────────────────────────────┐
│ Figure (the whole window/canvas)                         │
│  ┌────────────────────────────────────────────────────┐  │
│  │  Axes (the plotting area)                          │  │
│  │                                                    │  │
│  │   100 ┤          *                                 │  │
│  │      │       *     *                               │  │
│  │   80 ┤    *                                         │  │
│  │      │         *  *    *                            │  │
│  │   60 ┤  *                                         │  │
│  │      │              *    *                         │  │
│  │   40 ┤                   *                        │  │
│  │      └──┬──┬──┬──┬──┬──┬──┬──┬──┬──┬──            │  │
│  │         1  2  3  4  5  6  7  8  9  10  X-axis     │  │
│  └────────────────────────────────────────────────────┘  │
│  ┌─────────────────────┐  ┌────────────────────────┐    │
│  │  Axes 2             │  │  Axes 3                │    │
│  └─────────────────────┘  └────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Basic Plotting

```python
import matplotlib.pyplot as plt
import numpy as np

# ---- Line Plot ----
plt.figure(figsize=(10, 5))

x = np.linspace(0, 2 * np.pi, 100)
plt.plot(x, np.sin(x), 'b-', linewidth=2, label='sin(x)')
plt.plot(x, np.cos(x), 'r--', linewidth=2, label='cos(x)')
plt.plot(x, np.sin(x) * np.cos(x), 'g:', linewidth=2, label='sin(x)·cos(x)')

plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Trigonometric Functions', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('trig_functions.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Scatter Plots

```python
np.random.seed(42)

# Create clustered data
n_points = 200
clusters = [
    {'center': (2, 3), 'std': 0.5, 'label': 'Class A'},
    {'center': (7, 8), 'std': 0.8, 'label': 'Class B'},
    {'center': (5, 2), 'std': 0.6, 'label': 'Class C'},
]

fig, ax = plt.subplots(figsize=(8, 6))

for cluster in clusters:
    x = np.random.normal(cluster['center'][0], cluster['std'], n_points)
    y = np.random.normal(cluster['center'][1], cluster['std'], n_points)
    ax.scatter(x, y, alpha=0.6, s=30, label=cluster['label'])

ax.set_xlabel('Feature 1', fontsize=12)
ax.set_ylabel('Feature 2', fontsize=12)
ax.set_title('3-Class Scatter Plot', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.show()
```

### Bar Charts

```python
categories = ['Algorithm A', 'Algorithm B', 'Algorithm C', 'Algorithm D']
accuracy = [0.85, 0.91, 0.88, 0.93]
f1_score = [0.82, 0.89, 0.86, 0.91]
training_time = [120, 340, 200, 450]  # seconds

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Grouped bar chart
x_pos = np.arange(len(categories))
width = 0.35
axes[0].bar(x_pos - width/2, accuracy, width, label='Accuracy', color='steelblue')
axes[0].bar(x_pos + width/2, f1_score, width, label='F1-Score', color='coral')
axes[0].set_xlabel('Model')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison')
axes[0].set_xticks(x_pos)
axes[0].set_xticklabels(categories, rotation=15)
axes[0].legend()
axes[0].set_ylim(0.7, 1.0)

# Horizontal bar chart
colors = ['steelblue', 'coral', 'seagreen', 'orchid']
axes[1].barh(categories, training_time, color=colors, alpha=0.8)
axes[1].set_xlabel('Training Time (seconds)')
axes[1].set_title('Training Time by Algorithm')
for i, v in enumerate(training_time):
    axes[1].text(v + 5, i, f'{v}s', va='center')

plt.tight_layout()
plt.show()
```

### Histograms

```python
np.random.seed(42)
data_normal = np.random.normal(100, 15, 5000)
data_skewed = np.random.exponential(50, 5000)

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Basic histogram
axes[0].hist(data_normal, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[0].set_title('Normal Distribution (μ=100, σ=15)')
axes[0].set_xlabel('Value')
axes[0].set_ylabel('Frequency')

# Histogram with KDE
axes[1].hist(data_normal, bins=30, density=True, color='lightblue', alpha=0.5,
             edgecolor='black', label='Histogram')
x_kde = np.linspace(40, 160, 200)
kde = (1 / (15 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_kde - 100) / 15) ** 2)
axes[1].plot(x_kde, kde, 'r-', linewidth=2, label='PDF')
axes[1].set_title('Histogram + Theoretical PDF')
axes[1].legend()

# Comparison histogram
axes[2].hist(data_normal, bins=30, alpha=0.5, label='Normal', density=True)
axes[2].hist(data_skewed, bins=30, alpha=0.5, label='Exponential', density=True)
axes[2].set_title('Distribution Comparison')
axes[2].legend()

plt.tight_layout()
plt.show()
```

---

## Seaborn for Statistical Graphics

### Seaborn Setup and Themes

```python
import seaborn as sns

# Set theme for all plots
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# Available themes: darkgrid, whitegrid, dark, white, ticks
# Available palettes: deep, muted, bright, pastel, dark, colorblind

# Load built-in datasets
tips = sns.load_dataset('tips')
iris = sns.load_dataset('iris')
diamonds = sns.load_dataset('diamonds')  # May need internet

print("Tips dataset shape:", tips.shape)
print(tips.head())
```

### Distribution Plots

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histogram with KDE
sns.histplot(tips['total_bill'], kde=True, ax=axes[0, 0],
             color='steelblue', bins=25)
axes[0, 0].set_title('Histogram with KDE')

# KDE plot by category
sns.kdeplot(data=tips, x='total_bill', hue='time', fill=True,
            alpha=0.4, ax=axes[0, 1])
axes[0, 1].set_title('KDE by Time of Day')

# Box plot
sns.boxplot(data=tips, x='day', y='total_bill', hue='sex',
            ax=axes[1, 0])
axes[1, 0].set_title('Box Plot: Bill by Day & Gender')

# Violin plot
sns.violinplot(data=tips, x='day', y='total_bill', hue='sex',
               split=True, ax=axes[1, 1])
axes[1, 1].set_title('Violin Plot: Split by Gender')

plt.tight_layout()
plt.show()
```

### Box Plot Anatomy

```
           ┌───── Maximum (Q3 + 1.5*IQR)
           │
        ┌──┴──┐
        │     │         ← Upper Quartile (Q3 / 75th percentile)
        │     │
   ─────┤     ├─────    ← Median (Q2 / 50th percentile)
        │     │
        │     │         ← Lower Quartile (Q1 / 25th percentile)
        └──┬──┘
           │
           └───── Minimum (Q1 - 1.5*IQR)
              │
              ○         ← Outlier (beyond 1.5*IQR)
              ○

  IQR = Q3 - Q1

  Mathematical Definition:
  - Whisker length = 1.5 × IQR
  - Upper fence = Q3 + 1.5 × IQR
  - Lower fence = Q1 - 1.5 × IQR
  - Outlier: x < Q1 - 1.5×IQR OR x > Q3 + 1.5×IQR
```

---

## Relationship Plots

### Scatter and Regression Plots

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Basic scatter with regression
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0, 0],
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'})
axes[0, 0].set_title('Regression: Tip vs Total Bill')

# Scatter by category
sns.scatterplot(data=tips, x='total_bill', y='tip', hue='time',
                size='size', ax=axes[0, 1], alpha=0.7)
axes[0, 1].set_title('Scatter by Time & Party Size')

# Pair plot (for small number of features)
# sns.pairplot(iris, hue='species')  # Uncomment for full pairplot
# Instead, show a corner version
iris_subset = iris[['sepal_length', 'petal_length', 'species']]
g = sns.PairGrid(iris_subset, hue='species')
g.map_upper(sns.scatterplot, alpha=0.5)
g.map_lower(sns.kdeplot, fill=True, alpha=0.3)
g.map_diag(sns.histplot, kde=True)
g.add_legend()
# Note: PairGrid needs its own figure

# Joint plot (distribution + relationship)
sns.jointplot(data=tips, x='total_bill', y='tip',
              kind='reg', ax=axes[1, 0])  # kind: scatter, kde, hex, reg

# Residual plot
sns.residplot(data=tips, x='total_bill', y='tip',
              lowess=True, ax=axes[1, 1])
axes[1, 1].set_title('Residual Plot')
axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
```

### Heatmap for Correlation Matrix

```python
# Compute correlation matrix
corr_matrix = tips.select_dtypes(include=[np.number]).corr()

# Mathematical formula for Pearson correlation:
# r(x, y) = Σ(xᵢ - x̄)(yᵢ - ȳ) / √[Σ(xᵢ - x̄)² × Σ(yᵢ - ȳ)²]
# r ∈ [-1, 1], where -1 = perfect negative, 0 = no correlation, 1 = perfect positive

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix,
            annot=True,           # Show values
            fmt='.2f',            # 2 decimal places
            cmap='RdBu_r',        # Diverging colormap
            center=0,             # Center at zero
            square=True,          # Square cells
            linewidths=0.5,       # Cell borders
            cbar_kws={'shrink': 0.8},
            ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

---

## Categorical Plots

### Count, Bar, and Swarm Plots

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Count plot
sns.countplot(data=tips, x='day', hue='time', ax=axes[0, 0])
axes[0, 0].set_title('Count of Customers by Day & Time')

# Bar plot with confidence intervals
sns.barplot(data=tips, x='day', y='total_bill', hue='sex',
            estimator=np.mean, ci=95, ax=axes[0, 1])
axes[0, 1].set_title('Average Bill by Day & Gender (95% CI)')
# The CI bars represent: x̄ ± z_(α/2) × (s / √n)

# Point plot (shows trend)
sns.pointplot(data=tips, x='day', y='total_bill', hue='sex',
              markers=['o', 's'], linestyles=['-', '--'], ax=axes[1, 0])
axes[1, 0].set_title('Point Plot: Trend by Day')

# Swarm plot (shows individual points)
sns.swarmplot(data=tips, x='day', y='tip', hue='sex',
              size=4, ax=axes[1, 1])
axes[1, 1].set_title('Swarm Plot: Tip Distribution by Day')

plt.tight_layout()
plt.show()
```

### Faceting with Seaborn

```python
# FacetGrid: create a grid of plots by categories
g = sns.FacetGrid(tips, col='time', row='smoker', margin_titles=True,
                  height=4, aspect=1.2)
g.map_dataframe(sns.histplot, x='total_bill', kde=True, bins=20)
g.set_axis_labels('Total Bill ($)', 'Count')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.fig.suptitle('Bill Distribution by Time & Smoker Status', y=1.02)
plt.show()

# Catplot: one-liner categorical plot with faceting
sns.catplot(data=tips, x='day', y='total_bill',
            hue='sex', col='time',
            kind='box', height=5, aspect=0.8)
plt.suptitle('Catplot: Bill by Day, Gender, Time', y=1.02)
plt.show()
```

---

## Time Series Visualization

### Line Plots for Time Series

```python
import pandas as pd

# Generate sample time series
np.random.seed(42)
date_range = pd.date_range('2023-01-01', '2024-12-31', freq='D')
n = len(date_range)

trend = np.linspace(50, 80, n)
seasonal = 15 * np.sin(2 * np.pi * np.arange(n) / 365)
noise = np.random.normal(0, 5, n)
values = trend + seasonal + noise

ts = pd.DataFrame({
    'date': date_range,
    'value': values,
    'trend': trend,
    'predicted': trend + seasonal  # Perfect model prediction
}).set_index('date')

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# Raw time series
axes[0].plot(ts.index, ts['value'], linewidth=0.8, alpha=0.7, color='steelblue')
axes[0].set_title('Raw Time Series with Trend & Seasonality')
axes[0].set_ylabel('Value')
axes[0].grid(True, alpha=0.3)

# Moving averages
axes[1].plot(ts.index, ts['value'], linewidth=0.5, alpha=0.4, label='Daily', color='lightblue')
axes[1].plot(ts['value'].rolling(7).mean(), linewidth=1, label='7-day MA', color='blue')
axes[1].plot(ts['value'].rolling(30).mean(), linewidth=1.5, label='30-day MA', color='red')
axes[1].plot(ts['value'].rolling(90).mean(), linewidth=2, label='90-day MA', color='darkred')
axes[1].set_title('Moving Averages (Smoothing)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Actual vs Predicted
axes[2].plot(ts.index, ts['value'], linewidth=0.5, alpha=0.4, label='Actual', color='blue')
axes[2].plot(ts['predicted'], linewidth=1.5, label='Predicted', color='red')
axes[2].fill_between(ts.index, ts['value'], ts['predicted'],
                     alpha=0.2, color='orange', label='Residual')
axes[2].set_title('Actual vs Predicted (with Residuals)')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Seasonal Decomposition Visualization

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose time series
# Mathematical model: Y(t) = T(t) + S(t) + R(t)
# Where T = Trend, S = Seasonal, R = Residual
decomposition = seasonal_decompose(ts['value'], model='additive', period=365)

fig, axes = plt.subplots(4, 1, figsize=(14, 12))
decomposition.observed.plot(ax=axes[0], title='Observed')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Heatmaps & Correlation

### Custom Heatmaps

```python
# Create a confusion matrix visualization
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Use iris dataset
from sklearn.datasets import load_iris
iris_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris_data.data, iris_data.target, test_size=0.3, random_state=42
)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
class_names = iris_data.target_names

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion matrix heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# Normalized confusion matrix
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names,
            ax=axes[1])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')
axes[1].set_title('Normalized Confusion Matrix')

plt.tight_layout()
plt.show()

# Print classification report
print(classification_report(y_test, y_pred, target_names=class_names))
```

---

## Advanced Customization

### Color Theory for Data Visualization

```
┌──────────────────────────────────────────────────────────┐
│              Choosing the Right Colormap                  │
│                                                            │
│  Sequential (ordered data):                               │
│    Blues, Greens, Oranges, Viridis, Plasma               │
│    Use for: heatmaps of magnitude, density                │
│                                                            │
│  Diverging (deviation from center):                       │
│    RdBu, coolwarm, PiYG, BrBG, seismic                   │
│    Use for: correlation, positive/negative values         │
│                                                            │
│  Qualitative (categorical):                               │
│    Set1, Set2, Paired, tab10, Dark2                      │
│    Use for: categories with no inherent order             │
│                                                            │
│  Colorblind-friendly:                                     │
│    viridis, plasma, inferno, cividis                     │
│                                                            │
│  Avoid:                                                   │
│    ✗ Rainbow (uneven perceptual weight)                  │
│    ✗ Jet (creates false boundaries)                       │
│    ✗ Red-green (common colorblindness)                    │
└──────────────────────────────────────────────────────────┘
```

### Custom Styling

```python
# Comprehensive styling example
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
})

fig, ax = plt.subplots(figsize=(10, 6))

# Generate data
x = np.linspace(0, 10, 100)
for i, (color, label) in enumerate(zip(['#e74c3c', '#3498db', '#2ecc71', '#9b59b6'],
                                        ['Model A', 'Model B', 'Model C', 'Model D'])):
    y = 1 - np.exp(-0.3 * x) + np.random.normal(0, 0.03, len(x))
    y = np.clip(y, 0, 1)
    ax.plot(x, y, color=color, linewidth=2, label=label, alpha=0.8)

ax.set_xlabel('Training Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Model Training Curves', fontweight='bold', pad=15)
ax.legend(frameon=True, fancybox=True, shadow=True, loc='lower right')
ax.set_ylim(0, 1.05)

# Add annotation
ax.annotate('Convergence point',
            xy=(8, 0.95), xytext=(6, 0.7),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1.5),
            fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('training_curves.png', dpi=150, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.show()
```

### Custom Colormaps

```python
from matplotlib.colors import LinearSegmentedColormap

# Create custom colormap
colors_list = ['#003f5c', '#2f4b7c', '#665191', '#a05195',
               '#d45087', '#f95d6a', '#ff7c43', '#ffa600']
custom_cmap = LinearSegmentedColormap.from_list('custom', colors_list, N=256)

# Use in heatmap
data = np.random.randn(10, 10)
fig, ax = plt.subplots(figsize=(8, 7))
sns.heatmap(data, cmap=custom_cmap, annot=True, fmt='.1f',
            center=0, square=True, ax=ax)
ax.set_title('Custom Colormap Heatmap')
plt.tight_layout()
plt.show()
```

---

## Dashboard & Multi-Panel Layouts

### GridSpec for Complex Layouts

```python
from matplotlib.gridspec import GridSpec

fig = plt.figure(figsize=(16, 10))
fig.suptitle('ML Model Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.3)

# Panel 1: Feature distributions (top-left, spans 2 columns)
ax1 = fig.add_subplot(gs[0, :2])
np.random.seed(42)
feature_data = np.random.exponential(2, 1000)
ax1.hist(feature_data, bins=40, color='steelblue', alpha=0.7, edgecolor='white')
ax1.set_title('Feature Distribution')
ax1.set_xlabel('Value')
ax1.set_ylabel('Frequency')

# Panel 2: Metrics summary (top-right)
ax2 = fig.add_subplot(gs[0, 2])
ax2.axis('off')
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
values = [0.94, 0.92, 0.91, 0.915, 0.97]
for i, (metric, value) in enumerate(zip(metrics, values)):
    ax2.text(0.1, 0.85 - i * 0.17, f'{metric}: {value:.3f}',
             fontsize=12, transform=ax2.transAxes,
             fontweight='bold' if value == max(values) else 'normal')
ax2.set_title('Model Metrics')

# Panel 3: Correlation heatmap (middle-left, spans 2 rows)
ax3 = fig.add_subplot(gs[1:, 0])
corr = np.random.uniform(-1, 1, (5, 5))
np.fill_diagonal(corr, 1)
corr = (corr + corr.T) / 2  # Make symmetric
feature_names = [f'F{i}' for i in range(5)]
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=feature_names, yticklabels=feature_names,
            ax=ax3, square=True, cbar_kws={'shrink': 0.6})
ax3.set_title('Feature Correlation')

# Panel 4: Learning curves (middle-center)
ax4 = fig.add_subplot(gs[1, 1])
epochs = np.arange(1, 51)
train_loss = 2.0 * np.exp(-0.08 * epochs) + np.random.normal(0, 0.05, len(epochs))
val_loss = 2.0 * np.exp(-0.06 * epochs) + np.random.normal(0, 0.1, len(epochs))
ax4.plot(epochs, train_loss, label='Train Loss', color='steelblue')
ax4.plot(epochs, val_loss, label='Val Loss', color='coral')
ax4.set_title('Learning Curves')
ax4.legend()
ax4.set_xlabel('Epoch')

# Panel 5: ROC Curve (middle-right)
ax5 = fig.add_subplot(gs[1, 2])
fpr = np.linspace(0, 1, 100)
tpr = 1 - (1 - fpr) ** 3  # Simulated good ROC
tpr_noise = fpr + np.random.normal(0, 0.02, len(fpr))
ax5.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = 0.97')
ax5.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
ax5.fill_between(fpr, tpr, alpha=0.1, color='blue')
ax5.set_xlabel('False Positive Rate')
ax5.set_ylabel('True Positive Rate')
ax5.set_title('ROC Curve')
ax5.legend()

# Panel 6: Feature importance (bottom-center, spans 2 columns)
ax6 = fig.add_subplot(gs[2, 1:])
features = ['feature_5', 'feature_2', 'feature_8', 'feature_1',
            'feature_3', 'feature_7', 'feature_4', 'feature_6']
importances = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.07, 0.05]
colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features)))
ax6.barh(features, importances, color=colors)
ax6.set_xlabel('Importance')
ax6.set_title('Feature Importance (Random Forest)')

plt.savefig('ml_dashboard.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## Interactive Visualization

### Using Matplotlib Widgets

```python
from matplotlib.widgets import Slider

# Interactive function plotter
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)

x = np.linspace(-10, 10, 500)
line, = ax.plot(x, np.sin(x), linewidth=2)
ax.set_ylim(-3, 3)
ax.grid(True, alpha=0.3)

# Add sliders
ax_freq = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_amp = plt.axes([0.15, 0.05, 0.65, 0.03])
ax_phase = plt.axes([0.15, 0.15, 0.65, 0.03])

s_freq = Slider(ax_freq, 'Frequency', 0.1, 10.0, valinit=1.0)
s_amp = Slider(ax_amp, 'Amplitude', 0.1, 3.0, valinit=1.0)
s_phase = Slider(ax_phase, 'Phase', 0, 2 * np.pi, valinit=0)

def update(val):
    freq = s_freq.val
    amp = s_amp.val
    phase = s_phase.val
    line.set_ydata(amp * np.sin(freq * x + phase))
    fig.canvas.draw_idle()

s_freq.on_changed(update)
s_amp.on_changed(update)
s_phase.on_changed(update)

plt.show()
```

---

## Visualization Best Practices

### Chart Selection Guide

```
┌──────────────────────────────────────────────────────────┐
│                Choosing the Right Chart                    │
│                                                            │
│  What do you want to show?                                │
│                                                            │
│  ┌─ Comparison ─────────────────────────────────────┐    │
│  │ • Categorical → Bar chart, Grouped bar           │    │
│  │ • Time-based → Line chart, Area chart            │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
│  ┌─ Distribution ───────────────────────────────────┐    │
│  │ • Single variable → Histogram, KDE, Box plot     │    │
│  │ • By category → Violin, Ridge, Faceted hist      │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
│  ┌─ Relationship ──────────────────────────────────┐    │
│  │ • Two numeric → Scatter, Hexbin, 2D KDE         │    │
│  │ • Mixed types → Box, Violin, Bar with CI        │    │
│  └──────────────────────────────────────────────────┘    │
│                                                            │
│  ┌─ Composition ───────────────────────────────────┐    │
│  │ • Parts of whole → Pie (few), Stacked bar       │    │
│  │ • Over time → Stacked area, 100% stacked        │    │
│  └──────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────┘
```

### Common Mistakes to Avoid

```
DON'T:
  ✗ Truncate Y-axis to exaggerate differences
  ✗ Use 3D bar charts (perspective distorts values)
  ✗ Use dual Y-axes with different scales
  ✗ Encode too many variables in one chart
  ✗ Use rainbow color maps for sequential data
  ✗ Make pie charts with too many slices
  ✗ Omit axis labels or units

DO:
  ✓ Start Y-axis at zero for bar charts
  ✓ Use consistent scales when comparing
  ✓ Include clear titles and labels
  ✓ Use colorblind-friendly palettes
  ✓ Add annotations for key insights
  ✓ Keep it simple — remove chartjunk
  ✓ Use white/light backgrounds for readability
```

---

## Exercises with Solutions

### Exercise 1: EDA Visualization

```python
"""
Problem: Perform exploratory data analysis on the Tips dataset.
Create a comprehensive 2x3 grid of visualizations showing:
1. Distribution of total_bill
2. Tip vs total_bill scatter with regression
3. Average tip by day (bar chart)
4. Box plot of total_bill by time
5. Pairwise relationships of numeric columns
6. Count of smokers vs non-smokers by day
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

tips = sns.load_dataset('tips')

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Tips Dataset - Exploratory Data Analysis', fontsize=16, fontweight='bold')

# 1. Distribution of total_bill
sns.histplot(tips['total_bill'], kde=True, ax=axes[0, 0], color='steelblue')
axes[0, 0].set_title('Distribution of Total Bill')

# 2. Scatter with regression
sns.regplot(data=tips, x='total_bill', y='tip', ax=axes[0, 1],
            scatter_kws={'alpha': 0.5, 'color': 'steelblue'})
axes[0, 1].set_title('Tip vs Total Bill')

# 3. Average tip by day
daily_avg = tips.groupby('day')['tip'].mean().sort_values(ascending=False)
axes[0, 2].bar(daily_avg.index, daily_avg.values,
               color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'])
axes[0, 2].set_title('Average Tip by Day')

# 4. Box plot by time
sns.boxplot(data=tips, x='time', y='total_bill', ax=axes[1, 0],
            palette='Set2')
axes[1, 0].set_title('Total Bill Distribution by Time')

# 5. Correlation heatmap
numeric_cols = tips.select_dtypes(include=[np.number])
corr = numeric_cols.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, ax=axes[1, 1], square=True, cbar_kws={'shrink': 0.6})
axes[1, 1].set_title('Correlation Matrix')

# 6. Smoker count by day
sns.countplot(data=tips, x='day', hue='smoker', ax=axes[1, 2],
              palette='Set1')
axes[1, 2].set_title('Smoker Count by Day')

plt.tight_layout()
plt.savefig('tips_eda.png', dpi=150, bbox_inches='tight')
plt.show()
```

### Exercise 2: Model Performance Visualization

```python
"""
Problem: Given training history from a neural network, visualize:
1. Training and validation loss curves
2. Training and validation accuracy curves
3. Learning rate schedule
"""

np.random.seed(42)
epochs = np.arange(1, 101)

# Simulated training history
train_loss = 2.5 * np.exp(-0.04 * epochs) + np.random.normal(0, 0.02, len(epochs))
val_loss = 2.5 * np.exp(-0.035 * epochs) + np.random.normal(0, 0.05, len(epochs))
val_loss[70:] += 0.05 * np.arange(1, 31)  # Simulate overfitting

train_acc = 1 - 0.95 * np.exp(-0.05 * epochs) + np.random.normal(0, 0.005, len(epochs))
val_acc = 1 - 0.95 * np.exp(-0.04 * epochs) + np.random.normal(0, 0.01, len(epochs))
val_acc = np.clip(val_acc, 0, 1)
val_acc[70:] -= 0.003 * np.arange(1, 31)

lr = np.where(epochs < 30, 0.001,
              np.where(epochs < 60, 0.0005,
                       np.where(epochs < 80, 0.0001, 0.00005)))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Loss curves
axes[0].plot(epochs, train_loss, label='Train Loss', linewidth=2, color='steelblue')
axes[0].plot(epochs, val_loss, label='Val Loss', linewidth=2, color='coral')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (Cross-Entropy)')
axes[0].set_title('Loss Curves')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
# Mark best epoch
best_epoch = np.argmin(val_loss) + 1
axes[0].axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7)
axes[0].annotate(f'Best: Epoch {best_epoch}', xy=(best_epoch, val_loss[best_epoch-1]),
                 xytext=(best_epoch + 10, val_loss[best_epoch-1] + 0.2),
                 arrowprops=dict(arrowstyle='->', color='green'))

# Accuracy curves
axes[1].plot(epochs, train_acc, label='Train Accuracy', linewidth=2, color='steelblue')
axes[1].plot(epochs, val_acc, label='Val Accuracy', linewidth=2, color='coral')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('Accuracy Curves')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Learning rate schedule
axes[2].step(epochs, lr, where='mid', linewidth=2, color='purple')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Learning Rate')
axes[2].set_title('Learning Rate Schedule')
axes[2].set_yscale('log')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"Best model at epoch {best_epoch}")
print(f"  Train Loss: {train_loss[best_epoch-1]:.4f}, Val Loss: {val_loss[best_epoch-1]:.4f}")
```

### Exercise 3: PCA Visualization

```python
"""
Problem: Apply PCA to the Iris dataset and visualize:
1. 2D scatter plot of first two principal components
2. Explained variance ratio bar chart
3. 3D scatter plot of first three components
"""

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Standardize: z = (x - μ) / σ
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA: X = ZΣVᵀ
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Mathematical background:
# PCA finds eigenvectors of covariance matrix: C = (1/n)XᵀX
# Eigenvalues λ represent variance captured
# Explained variance ratio: λᵢ / Σλⱼ

fig = plt.figure(figsize=(16, 5))

# 2D PCA scatter
ax1 = fig.add_subplot(131)
colors = ['#e74c3c', '#3498db', '#2ecc71']
for i, target_name in enumerate(target_names):
    ax1.scatter(X_pca[y == i, 0], X_pca[y == i, 1],
                color=colors[i], alpha=0.7, label=target_name, edgecolors='white', s=60)
ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax1.set_title('PCA: First Two Components')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Explained variance
ax2 = fig.add_subplot(132)
cumulative = np.cumsum(pca.explained_variance_ratio_)
ax2.bar(range(1, 5), pca.explained_variance_ratio_ * 100, alpha=0.7,
        color='steelblue', label='Individual')
ax2.step(range(1, 5), cumulative * 100, where='mid', color='red',
         label='Cumulative', linewidth=2)
ax2.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% threshold')
ax2.set_xlabel('Principal Component')
ax2.set_ylabel('Explained Variance (%)')
ax2.set_title('Explained Variance Ratio')
ax2.legend()
ax2.set_xticks(range(1, 5))
ax2.set_xticklabels([f'PC{i}' for i in range(1, 5)])

# 3D scatter
ax3 = fig.add_subplot(133, projection='3d')
for i, target_name in enumerate(target_names):
    ax3.scatter(X_pca[y == i, 0], X_pca[y == i, 1], X_pca[y == i, 2],
                color=colors[i], alpha=0.7, label=target_name, s=60)
ax3.set_xlabel('PC1')
ax3.set_ylabel('PC2')
ax3.set_zlabel('PC3')
ax3.set_title('PCA: 3D View')

plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=150, bbox_inches='tight')
plt.show()

print("Explained variance ratios:", pca.explained_variance_ratio_)
print("Cumulative variance:", cumulative)
```

---

## ML Connections

### Visualization in the ML Lifecycle

```
┌──────────────────────────────────────────────────────────────┐
│                Visualization in ML Workflow                    │
│                                                               │
│  Phase              Visualization Types                      │
│  ──────────────────────────────────────────────              │
│  Data Exploration   Histograms, scatter plots, pair plots    │
│  Feature Eng.       Distributions, correlation heatmaps      │
│  Model Training     Learning curves, loss plots              │
│  Model Evaluation   Confusion matrix, ROC, PR curves         │
│  Error Analysis     Residual plots, feature importance       │
│  Model Comparison   Bar charts, radar charts                 │
│  Deployment         Dashboards, monitoring plots             │
│                                                               │
│  Key Visualizations for Model Selection:                      │
│  ├── ROC Curve: TPR vs FPR at thresholds                     │
│  │   AUC = ∫₀¹ TPR(FPR) dFPR                               │
│  ├── PR Curve: Precision vs Recall                           │
│  │   AP = Σₙ (Rₙ - Rₙ₋₁) × Pₙ                             │
│  ├── Learning Curve: Score vs Training Size                  │
│  │   Detect bias/variance tradeoff                           │
│  └── Validation Curve: Score vs Hyperparameter               │
│      Find optimal hyperparameter values                      │
└──────────────────────────────────────────────────────────────┘
```

### Dimensionality Reduction Visualization

```python
# Compare PCA vs t-SNE vs UMAP (conceptual code)
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE

digits = load_digits()
X, y = digits.data, digits.target

# t-SNE: minimizes KL divergence between distributions
# P_ji ∝ exp(-||xᵢ - xⱼ||² / 2σᵢ²)
# Q_ji ∝ exp(-||yᵢ - yⱼ||²)
# Loss = KL(P || Q) = Σᵢ Σⱼ P_ji log(P_ji / Q_ji)

tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# t-SNE
scatter = axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y,
                          cmap='tab10', alpha=0.6, s=15)
axes[0].set_title('t-SNE Visualization (Digits Dataset)')
plt.colorbar(scatter, ax=axes[0], label='Digit')

# PCA comparison
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y,
                           cmap='tab10', alpha=0.6, s=15)
axes[1].set_title(f'PCA Visualization ({pca.explained_variance_ratio_.sum()*100:.1f}% var)')
plt.colorbar(scatter2, ax=axes[1], label='Digit')

plt.tight_layout()
plt.show()
```

### SHAP Values Visualization

```python
"""
SHAP (SHapley Additive exPlanations) values explain individual predictions.
f(x) = φ₀ + Σᵢ φᵢ
Where φ₀ is the base value and φᵢ is the contribution of feature i.
"""

# Conceptual SHAP summary plot
np.random.seed(42)
n_samples = 200
n_features = 5
feature_names = ['Age', 'Income', 'Score', 'Balance', 'Tenure']

# Simulated SHAP values
shap_values = np.random.randn(n_samples, n_features) * np.array([0.3, 0.5, 0.8, 0.4, 0.2])
feature_values = np.random.randn(n_samples, n_features)

fig, ax = plt.subplots(figsize=(10, 5))

for i in range(n_features):
    color = feature_values[:, i]
    scatter = ax.scatter(shap_values[:, i], np.full(n_samples, i),
                        c=color, cmap='RdBu_r', alpha=0.5, s=15,
                        vmin=-2, vmax=2)

ax.set_xlabel('SHAP Value (impact on model output)')
ax.set_yticks(range(n_features))
ax.set_yticklabels(feature_names)
ax.set_title('SHAP Summary Plot (conceptual)')
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.2)
cbar = plt.colorbar(scatter, ax=ax, label='Feature Value', shrink=0.8)
plt.tight_layout()
plt.show()
```

---

## Summary

This guide covered the full spectrum of data visualization from basic plotting to advanced dashboards:

1. **Matplotlib** provides the foundation — understand Figure/Axes architecture
2. **Seaborn** simplifies statistical visualization with high-level functions
3. **Distribution plots** (histogram, KDE, box, violin) reveal data characteristics
4. **Relationship plots** (scatter, reg, joint) show variable correlations
5. **Time series** require special handling — decomposition, moving averages
6. **Dashboards** use GridSpec for complex multi-panel layouts
7. **ML visualization** includes learning curves, ROC, confusion matrices, SHAP
