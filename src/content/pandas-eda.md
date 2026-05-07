# Pandas & Exploratory Data Analysis (EDA) on Datasets

## Why Pandas Matters in AI/ML

Before you train any model, you must understand your data. This process — Exploratory Data Analysis (EDA) — typically consumes 60-80% of a data scientist's time. Pandas is the tool that makes this possible.

Pandas was created by Wes McKinney in 2008 at AQR Capital Management (a quantitative hedge fund) because existing tools could not handle the messy, heterogeneous, real-world data that financial analysts work with daily. Today, every Kaggle competition notebook, every ML pipeline, and every data science workflow starts with `import pandas as pd`.

The core data structures are:
- **Series**: A single column of data with an index (like a labeled array)
- **DataFrame**: A 2D table with labeled rows and columns (like a spreadsheet or SQL table)

```
            Column A    Column B    Column C
Index 0       10          20          30
Index 1       40          50          60
Index 2       70          80          90
         ↓            ↓            ↓
      Series       Series       Series
              ↓
           DataFrame
```

---

## 1. Creating DataFrames and Series

```python
import pandas as pd
import numpy as np

# From dictionary — the most common way
df = pd.DataFrame({
    "feature_1": [1.2, 3.4, 5.6, 7.8, 9.0],
    "feature_2": [10, 20, 30, 40, 50],
    "label": ["A", "B", "A", "C", "B"]
})
print(df)

# From NumPy array
data = np.random.randn(100, 5)
columns = [f"sensor_{i}" for i in range(1, 6)]
df = pd.DataFrame(data, columns=columns)

# Series
s = pd.Series([10, 20, 30, 40], index=["a", "b", "c"], name="values")
print(s.mean())          # 20.0
print(s.value_counts())  # count of unique values

# Loading real data
titanic = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
print(titanic.shape)    # (891, 12)
print(titanic.columns)
```

---

## 2. First Look at Data — The EDA Checklist

Every time you receive a new dataset, run this sequence:

```python
df = titanic.copy()

# 1. Shape — how much data?
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# 2. First/last rows
print(df.head(10))
print(df.tail(5))

# 3. Data types and non-null counts
print(df.info())
# PassengerId    891 non-null    int64
# Survived       891 non-null    int64
# Age            714 non-null    float64   ← missing values!
# Cabin          204 non-null    object    ← lots of missing!
# ...

# 4. Summary statistics — the single most important EDA command
print(df.describe())
#              count    mean     std    min     25%     50%     75%     max
# PassengerId  891.0  446.0   257.4   1.0   223.5   446.0   668.5   891.0
# Survived     891.0    0.38    0.49   0.0     0.0     0.0     1.0     1.0
# Age          714.0   29.7    14.5   0.42   20.1    28.0    38.0    80.0
# Fare         891.0   32.2    49.7   0.0     7.9    14.5    31.0   512.3

# 5. Missing values — critical for ML
print(df.isnull().sum())
# Age         177
# Cabin       687
# Embarked      2

# 6. Unique values per column — for categorical columns
for col in df.select_dtypes(include="object").columns:
    print(f"{col}: {df[col].nunique()} unique values")

# 7. Distribution of target variable — check for class imbalance
print(df["Survived"].value_counts(normalize=True))
# 0    0.616   (did not survive)
# 1    0.384   (survived)
```

---

## 3. Data Cleaning — The Real Work

### 3.1 Handling Missing Values

```python
# Strategies for missing data
df = titanic.copy()

# Strategy 1: Drop rows with missing values
df_clean = df.dropna(subset=["Embarked"])  # Only 2 missing — safe to drop

# Strategy 2: Fill with a constant
df["Cabin_known"] = df["Cabin"].notna().astype(int)  # Useful feature: did they have a cabin?

# Strategy 3: Fill with statistics
age_median = df["Age"].median()  # Robust to outliers
df["Age_filled"] = df["Age"].fillna(age_median)

# Strategy 4: Forward/backward fill (time series data)
# df["temperature"] = df["temperature"].fillna(method="ffill")

# Strategy 5: Group-based imputation
df["Age"] = df.groupby(["Pclass", "Sex"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
# Children in 3rd class get a different imputed age than adults in 1st class

print(f"Missing after cleaning: {df[['Age', 'Embarked']].isnull().sum().sum()}")
```

### 3.2 Handling Duplicates

```python
# Check for duplicate rows
print(f"Duplicates: {df.duplicated().sum()}")

# Check for duplicate in specific columns (e.g., same passenger recorded twice)
print(f"Duplicate names: {df.duplicated(subset=['Name']).sum()}")

# Remove duplicates
df = df.drop_duplicates()
```

### 3.3 Type Conversions

```python
# Convert types for efficiency and correctness
df["Survived"] = df["Survived"].astype("category")   # Small integer → category
df["Pclass"] = df["Plass"].astype("category") if "Plass" in df.columns else df["Pclass"].astype("category")
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")  # Convert, invalid → NaN

# Convert string dates
df["DateColumn"] = pd.to_datetime(df["DateColumn"])

# Memory savings from proper types
print(f"Memory before: {titanic.memory_usage(deep=True).sum() / 1024:.1f} KB")
print(f"Memory after:  {df.memory_usage(deep=True).sum() / 1024:.1f} KB")
```

---

## 4. Selecting and Filtering Data

```python
df = titanic.copy()

# Select columns
print(df["Age"])                    # Single column → Series
print(df[["Age", "Fare", "Survived"]])  # Multiple columns → DataFrame

# Select by position (iloc — integer location)
print(df.iloc[0])            # First row
print(df.iloc[0:5, 1:4])     # Rows 0-4, columns 1-3

# Select by label (loc — label-based)
print(df.loc[0])                    # Row with index 0
print(df.loc[df.index[:5], ["Name", "Age", "Fare"]])

# Boolean filtering — the most common operation
print(df[df["Age"] > 70])                        # Passengers over 70
print(df[(df["Age"] > 30) & (df["Fare"] < 20)])  # Compound conditions
print(df[df["Name"].str.contains("Mrs."))]       # String search

# isin() — membership testing
df[df["Pclass"].isin([1, 2])]  # First and second class

# query() — SQL-like syntax for complex filters
print(df.query("Age > 30 and Fare < 20 and Survived == 1"))
```

### Advanced: `.loc` with conditions

```python
# Conditional modification — set Fare to NaN for free tickets
df.loc[df["Fare"] == 0, "Fare"] = np.nan

# Conditional creation of a new column
df.loc[df["Age"] < 18, "AgeGroup"] = "Child"
df.loc[(df["Age"] >= 18) & (df["Age"] < 60), "AgeGroup"] = "Adult"
df.loc[df["Age"] >= 60, "AgeGroup"] = "Senior"

print(df["AgeGroup"].value_counts())
```

---

## 5. Feature Engineering

Feature engineering is where domain expertise transforms raw data into ML-ready features. This is often the difference between a mediocre and a winning model.

```python
df = titanic.copy()

# Extract title from name
df["Title"] = df["Name"].str.extract(r" ([A-Za-z]+)\.")
print(df["Title"].value_counts())

# Group rare titles
rare_titles = df["Title"].value_counts()[df["Title"].value_counts() < 10].index
df["Title"] = df["Title"].replace(rare_titles, "Rare")

# Family size from SibSp and Parch
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1  # +1 for self

# Is alone?
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

# Binning continuous variables
df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 18, 35, 60, 100],
                       labels=["Child", "Teen", "Young Adult", "Adult", "Senior"])
df["FareBin"] = pd.qcut(df["Fare"], q=4, labels=["Low", "Medium", "High", "Very High"])

# Interaction features
df["FarePerPerson"] = df["Fare"] / df["FamilySize"]

# Log transform for skewed distributions
df["FareLog"] = np.log1p(df["Fare"])  # log(1+x) to handle zeros

print(df[["Title", "FamilySize", "IsAlone", "AgeBin", "FareBin", "FareLog"]].head(10))
```

### Encoding Categorical Variables

```python
# One-hot encoding
df_encoded = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
# Sex_male, Embarked_Q, Embarked_S (drop_first avoids multicollinearity)

# Label encoding (for ordinal variables)
age_order = {"Child": 0, "Teen": 1, "Young Adult": 2, "Adult": 3, "Senior": 4}
df["AgeBin_encoded"] = df["AgeBin"].map(age_order)

# Target encoding (mean of target per category — use with caution to avoid leakage)
target_means = df.groupby("Title")["Survived"].mean()
df["Title_encoded"] = df["Title"].map(target_means)
print(df.groupby("Title")["Survived"].mean())
```

---

## 6. GroupBy — Aggregations and Analysis

GroupBy is Pandas' equivalent of SQL `GROUP BY` — split data into groups, apply a function, combine results.

```python
df = titanic.copy()
df["Age"] = df["Age"].fillna(df["Age"].median())

# Basic groupby
print(df.groupby("Pclass")["Survived"].mean())
# Pclass
# 1    0.6296    (63% of 1st class survived)
# 2    0.4728
# 3    0.2424    (only 24% of 3rd class)

# Multiple aggregations
agg_result = df.groupby("Pclass").agg({
    "Age": ["mean", "median", "std"],
    "Fare": ["mean", "min", "max"],
    "Survived": ["mean", "sum", "count"]
})
print(agg_result)

# Group by multiple columns
survival_table = df.groupby(["Pclass", "Sex"])["Survived"].agg(
    ["mean", "count"]
).rename(columns={"mean": "Survival Rate", "count": "Passengers"})
print(survival_table)
#                     Survival Rate  Passengers
# Pclass Sex
# 1      female         0.968085          94
#        male           0.368852         122
# 2      female         0.921053          76
#        male           0.157407         108
# 3      female         0.500000         144
#        male           0.135447         347

# Transform — apply per-group and return same shape
df["Age_normalized"] = df.groupby("Pclass")["Age"].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Filter — keep only groups that meet a condition
large_families = df.groupby("Ticket").filter(lambda x: len(x) > 4)
print(f"Families with >4 members: {len(large_families)} passengers")
```

---

## 7. Merge, Join, and Concatenate

Real-world ML projects combine data from multiple sources. Pandas provides SQL-like join operations.

```python
# Sample datasets
students = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "name": ["Alice", "Bob", "Charlie", "Diana", "Eve"],
    "major": ["CS", "Math", "CS", "Physics", "Math"]
})

grades = pd.DataFrame({
    "student_id": [1, 2, 3, 6, 7],  # Note: students 4,5 missing; 6,7 are new
    "course": ["ML", "ML", "Stats", "ML", "Stats"],
    "grade": [95, 82, 88, 91, 79]
})

attendance = pd.DataFrame({
    "student_id": [1, 2, 3, 4, 5],
    "attendance_pct": [98, 85, 92, 75, 88]
})

# INNER JOIN — only matching rows
inner = pd.merge(students, grades, on="student_id", how="inner")
print(f"Inner join: {len(inner)} rows")  # 3 rows (student_ids 1, 2, 3)

# LEFT JOIN — all from left, matching from right
left = pd.merge(students, grades, on="student_id", how="left")
print(f"Left join: {len(left)} rows")  # 5 rows (student_ids 1-5, grades NaN for 4,5)

# OUTER JOIN — all from both
outer = pd.merge(students, grades, on="student_id", how="outer")
print(f"Outer join: {len(outer)} rows")  # 7 rows

# Merge on multiple columns
pd.merge(df1, df2, on=["key1", "key2"], how="inner")

# Concatenate — stack DataFrames vertically
df_combined = pd.concat([df_q1, df_q2, df_q3], axis=0, ignore_index=True)

# Merge multiple DataFrames at once (like a database JOIN chain)
full_data = students.merge(grades, on="student_id", how="left") \
                    .merge(attendance, on="student_id", how="left")
```

### Join Types Diagram

```
    Left Table        Right Table
    +------+          +------+
    |  1   |          |  1   |
    |  2   |          |  2   |
    |  3   |          |  3   |
    |  4   |          |  6   |
    |  5   |          |  7   |
    +------+          +------+

INNER:            LEFT:             OUTER:
+------+          +------+          +------+
|  1   |          |  1   |          |  1   |
|  2   |          |  2   |          |  2   |
|  3   |          |  3   |          |  3   |
+------+          |  4   |          |  4   |
                  |  5   |          |  5   |
                  +------+          |  6   |
                                    |  7   |
                                    +------+
```

---

## 8. Pivot Tables and Reshaping

```python
df = titanic.copy()

# Pivot table — survival rate by class and sex
pivot = pd.pivot_table(
    df,
    values="Survived",
    index="Pclass",
    columns="Sex",
    aggfunc="mean"
)
print(pivot)

# Pivot table with multiple aggregations
pivot_multi = pd.pivot_table(
    df,
    values=["Survived", "Fare", "Age"],
    index="Pclass",
    columns="Sex",
    aggfunc={"Survived": "mean", "Fare": ["mean", "std"], "Age": "median"}
)

# Melt — wide format to long format (unpivot)
wide_df = pd.DataFrame({
    "student": ["Alice", "Bob"],
    "math": [95, 82],
    "science": [88, 91],
    "english": [79, 85]
})
long_df = wide_df.melt(
    id_vars=["student"],
    value_vars=["math", "science", "english"],
    var_name="subject",
    value_name="score"
)
# student  subject  score
# Alice    math      95
# Alice    science   88
# Alice    english   79
# Bob      math      82
# Bob      science   91
# Bob      english   85

# Crosstab — frequency table
print(pd.crosstab(df["Pclass"], df["Survived"], margins=True))
```

---

## 9. Time Series with Pandas

```python
# Create a time series
dates = pd.date_range("2023-01-01", periods=365, freq="D")
values = np.sin(np.arange(365) * 2 * np.pi / 365) * 20 + 50 + np.random.randn(365) * 5
ts = pd.Series(values, index=dates, name="temperature")

# Resampling — change frequency
daily_avg = ts.resample("D").mean()     # Already daily
weekly_avg = ts.resample("W").mean()    # Weekly average
monthly_avg = ts.resample("ME").mean()  # Monthly average

# Rolling windows — moving statistics
rolling_mean = ts.rolling(window=7).mean()     # 7-day moving average
rolling_std = ts.rolling(window=7).std()       # 7-day moving std
exp_smooth = ts.ewm(span=7).mean()             # Exponential smoothing

# Shifting — create lag features for ML
ts_df = pd.DataFrame({"temperature": ts})
ts_df["temp_lag1"] = ts_df["temperature"].shift(1)
ts_df["temp_lag7"] = ts_df["temperature"].shift(7)
ts_df["temp_rolling_mean_7"] = ts_df["temperature"].rolling(7).mean()

# Date-based filtering
summer = ts["2023-06-01":"2023-08-31"]
```

---

## 10. Full EDA Example: Real Dataset

```python
import pandas as pd
import numpy as np

def perform_eda(filepath_or_url: str, target_col: str = None):
    """Comprehensive EDA workflow."""
    print("=" * 70)
    print("EXPLORATORY DATA ANALYSIS REPORT")
    print("=" * 70)

    # Load data
    df = pd.read_csv(filepath_or_url)
    print(f"\nDataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")

    # Section 1: Data Types
    print("-" * 70)
    print("1. DATA TYPES")
    print("-" * 70)
    print(df.dtypes)
    print(f"\nMemory Usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    # Section 2: Missing Values
    print(f"\n{'-' * 70}")
    print("2. MISSING VALUES")
    print("-" * 70)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({"Missing": missing, "Percentage": missing_pct})
    missing_df = missing_df[missing_df["Missing"] > 0].sort_values("Percentage", ascending=False)
    if len(missing_df) > 0:
        print(missing_df)
    else:
        print("No missing values found!")

    # Section 3: Numeric Summary
    print(f"\n{'-' * 70}")
    print("3. NUMERIC COLUMNS SUMMARY")
    print("-" * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(df[numeric_cols].describe().T)

    # Section 4: Categorical Summary
    print(f"\n{'-' * 70}")
    print("4. CATEGORICAL COLUMNS SUMMARY")
    print("-" * 70)
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        print(f"\n{col} ({df[col].nunique()} unique values):")
        print(df[col].value_counts().head(10))

    # Section 5: Target Distribution
    if target_col and target_col in df.columns:
        print(f"\n{'-' * 70}")
        print(f"5. TARGET DISTRIBUTION: {target_col}")
        print("-" * 70)
        print(df[target_col].value_counts())
        if df[target_col].dtype in [np.float64, np.int64]:
            print(f"\nSkewness: {df[target_col].skew():.3f}")
            print(f"Kurtosis: {df[target_col].kurtosis():.3f}")

    # Section 6: Correlations
    if len(numeric_cols) > 1:
        print(f"\n{'-' * 70}")
        print("6. TOP CORRELATIONS")
        print("-" * 70)
        corr_matrix = df[numeric_cols].corr()
        # Get top correlations (excluding self-correlation)
        corr_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                corr_pairs.append((
                    numeric_cols[i], numeric_cols[j], corr_matrix.iloc[i, j]
                ))
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        for col1, col2, corr in corr_pairs[:10]:
            print(f"  {col1:20s} vs {col2:20s}: {corr:+.4f}")

    # Section 7: Potential Issues
    print(f"\n{'-' * 70}")
    print("7. POTENTIAL ISSUES")
    print("-" * 70)
    if len(missing_df) > 0:
        print(f"  [!] {len(missing_df)} columns have missing values")
    # Check for duplicates
    n_dupes = df.duplicated().sum()
    if n_dupes > 0:
        print(f"  [!] {n_dupes} duplicate rows found")
    # Check for constant columns
    const_cols = [col for col in df.columns if df[col].nunique() == 1]
    if const_cols:
        print(f"  [!] Constant columns (no information): {const_cols}")
    # Check for high cardinality
    for col in cat_cols:
        if df[col].nunique() > 50:
            print(f"  [!] High cardinality in '{col}': {df[col].nunique()} unique values")

    print(f"\n{'=' * 70}")
    print("EDA COMPLETE")
    print("=" * 70)

# Run the full EDA
perform_eda(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    target_col="Survived"
)
```

---

## 11. Exercises

### Exercise 1: California Housing EDA

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target

# Tasks:
# 1. Find the top 3 features most correlated with MedHouseVal
# 2. Identify outliers in each column (values > 3 std from mean)
# 3. Create binned versions of MedInc and report average house value per bin
# 4. Find the zip-code-like grouping with highest average house value
#    (use latitude/longitude rounding as proxy for location)
```

### Exercise 2: Data Cleaning Pipeline

Build a reusable function that takes a DataFrame and:
1. Removes columns with >50% missing values
2. Imputes remaining missing values (median for numeric, mode for categorical)
3. Removes duplicate rows
4. Converts object columns to category where unique values < 10
5. Returns the cleaned DataFrame and a cleaning report

### Exercise 3: Customer Segmentation

Using a sales dataset, create:
- A pivot table showing revenue by product category and month
- A summary of top 10 customers by total spend
- A cohort analysis (first purchase month vs retention rate)

---

## Connections to Other ML Topics

| Pandas Operation | ML Application |
|-----------------|----------------|
| `df.describe()` | Detect outliers, understand feature ranges |
| `df.groupby()` | Stratified analysis, per-class statistics |
| `pd.get_dummies()` | One-hot encoding for categorical features |
| `df.merge()` | Joining features from multiple data sources |
| `df.pivot_table()` | Aggregating predictions by segments |
| Rolling windows | Time series feature engineering |
| Missing value handling | Prerequisite for any model training |
| `df.corr()` | Feature selection, multicollinearity detection |

---

## Key Takeaways

1. **`df.info()` and `df.describe()`** are your first two commands on any dataset.
2. **Missing values** must be handled before modeling — understand *why* data is missing.
3. **`df.groupby()`** reveals patterns that aggregate statistics hide (survival by class AND gender).
4. **Feature engineering** in pandas (binning, encoding, interactions) often matters more than model choice.
5. **Merges** let you combine features from multiple tables — common in production ML systems.
6. **Pivot tables** summarize multi-dimensional relationships in a single readable table.
7. **The EDA workflow** (load → inspect → clean → transform → analyze) is universal across all ML projects.
