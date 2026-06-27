## Module 2: Data Wrangling — Dealing with Missing Values in Python

Missing data is an inevitable challenge in real-world data science. A missing value condition occurs whenever an observation lacks an entry for a specific feature. In raw datasets, missing data rarely manifests uniformly; it frequently appears as question marks (`?`), character codes (`NA` or `NaN`), numerical placeholders (`0`), or completely blank cells.

Failing to systematically address these missing gaps can bias statistical summaries, corrupt calculations, and break downstream machine learning algorithms.

---

## Core Concept & "Why it Matters"

When handling missing entries, data professionals must weigh data retention against mathematical distortion. Every missing data strategy comes with distinct trade-offs:

### 1. Data Removal (Dropping)

* **Row Dropping (`axis=0`):** Eliminating the entire observation/sample. This is highly recommended when missing data is sparse or when the missing value resides in your **target/dependent label variable** (e.g., a car missing its `Price`).
* **Column Dropping (`axis=1`):** Eliminating the entire feature vector. This is used if a feature is missing a massive percentage of its entries, making it analytically useless.
* *Trade-off:* Minimizes data distortion but risks wasting valuable information contained within the remaining, non-null columns of that row.

### 2. Data Replacement (Imputation)

* **Numerical Features:** Imputing missing entries using the mathematical mean (average) or median of the entire column.
* **Categorical Features:** Imputing missing string entries using the mode (the most frequently occurring class, such as "Gasoline").
* **Domain Knowledge Imputation:** Injecting expert insight (e.g., knowing that unrecorded insurance risk values heavily correlate with historical vehicle age groups).
* *Trade-off:* Preserves sample size and statistical power, but introduces artificial approximations that can reduce dataset variance.

---

## The Python Execution

### 1. Dropping Rows with Missing Target Labels

When a row is missing the value you are actively trying to predict (the target variable, like `price`), imputing it with a guess introduces severe bias. It is standard practice to drop these specific rows completely.

```python
import pandas as pd
import numpy as np

# Load sample data with a missing target value
data = {'make': ['toyota', 'mazda', 'audi'], 'price': [12000, np.nan, 23000]}
df = pd.DataFrame(data)

# Drop rows where 'price' is missing
# axis=0 target rows; subset specifies which column to inspect
df.dropna(subset=["price"], axis=0, inplace=True)

print("DataFrame after dropping rows missing target labels:")
print(df)

```

### 2. Calculating Mean and Imputing/Replacing Numerical Gaps

For non-target features (like `normalized-losses`), calculating the column average and replacing the gaps allows you to maintain your sample size.

```python
# Sample data with missing numerical feature values
data = {'normalized-losses': [150, np.nan, 110, 130], 'price': [10000, 12000, 9000, 15000]}
df_cars = pd.DataFrame(data)

# 1. Calculate the mean of the column (Pandas automatically excludes NaNs)
mean_loss = df_cars['normalized-losses'].mean()
print(f"Calculated Column Mean: {mean_loss}")

# 2. Use the .replace() method to swap NaN elements with the calculated mean
# Note: Requires numpy's np.nan identifier to target missing flags
df_cars['normalized-losses'].replace(np.nan, mean_loss, inplace=True)

print("\nDataFrame after mean imputation:")
print(df_cars)

```

---

## Interview Cheat Sheet

### 💡 The "Must-Knows" (Technical Gotchas)

* **The `inplace=True` Mutation Pattern:** Methods like `.dropna()` and `.replace()` do not alter the underlying DataFrame by default; they generate and return a brand-new copy. To mutate the data object directly in RAM, you must set `inplace=True`. Alternatively, modern clean-coding guidelines often prioritize direct variable reassignment (`df = df.dropna()`) to maintain functional programming clarity.
* **The Hidden Data Type Upcast:** In Pandas, `np.nan` is technically classified as a floating-point data type (`float64`). If you have a clean integer column (`int64`) and a single missing value or `NaN` is introduced, **Pandas will automatically upcast the entire column data type to a float.**
* **The Pitfall of Imputing with the Global Mean:** While global mean imputation is a fast baseline approach, doing so across highly diverse structural sub-groups can mute real variances. An advanced candidate should mention that grouping features first (e.g., grouping by car `make` or `body-style` and calculating localized means) yields far more accurate imputations.

### ❓ Common Interview Questions

#### Q1: Why is it standard practice to drop rows missing their target labels rather than imputing them?

* **How to Answer:** The target label is the exact ground truth variable that an analytical model or machine learning algorithm attempts to learn or predict. If you impute missing target labels using a statistical guess (like the mean or mode), you are training your model on synthetic data. This introduces significant artificial bias and invalidates your evaluation metrics, as the model would be optimized to predict your own artificial assumptions rather than true real-world variations.

#### Q2: What is the mechanical difference between executing `df.dropna(inplace=True)` and `df = df.dropna()`? Which is generally preferred in modern pipelines?

* **How to Answer:** `df.dropna(inplace=True)` modifies the original DataFrame directly in place without creating a new object in memory. On the other hand, `df = df.dropna()` creates a brand-new modified copy of the DataFrame in memory and reassigns the variable name `df` to point to this new object. While `inplace=True` was historically favored for memory conservation, modern data engineering pipelines generally prefer explicit variable reassignment. This practice improves readability, prevents side-effects during method chaining, and avoids unexpected behavior down the road.

#### Q3: How do you handle missing values when dealing with categorical/text features instead of numerical ones?

* **How to Answer:** For categorical features, calculating a mathematical mean is impossible. The standard approach is to impute missing entries using the **mode** (the most frequently occurring category within that column). If the categorical column has a high percentage of missing values, or if the missingness itself contains predictive meaning, a more robust technique is to treat the missing gap as its own distinct category label by filling it with a placeholder string like `"Unknown"` or `"Missing"`.