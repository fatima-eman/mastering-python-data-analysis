## Module 1: Getting Started Analyzing Data in Python

Once a dataset is successfully loaded into a Pandas DataFrame, the immediate next phase is **Exploratory Data Analysis (EDA)**. Before building statistical models or running complex calculations, data scientists must profile the dataset to understand its structural composition, inspect its data types, and evaluate its statistical distributions.

Early data profiling acts as an early warning system to catch hidden anomalies—such as data type mismatches or extreme outliers—before they break downstream analysis pipelines.

---

## Core Concept & "Why it Matters"

### 1. Data Type Profiling: Python vs. Pandas

Pandas automatically infers data types based on the file encoding it detects during the import process. However, this inference engine can often misinterpret column structures. Checking data types is essential for two reasons:

1. **Ensuring Analytical Compatibility:** Mathematical functions can only be executed on numerical data types. Attempting to calculate a mean on an uncorrected data column will throw a runtime error.
2. **Identifying Data Mismatches:** If a column like `Car Price` or `Engine Bore` is parsed as a non-numeric format, it indicates corrupt or uncleaned entries within the original source file.

| Native Python Type | Pandas Dtype | Analytical Purpose |
| --- | --- | --- |
| `str` | **`object`** | Represents text data or mixed alphanumeric values. |
| `int` | **`int64`** | Represents discrete integer values. |
| `float` | **`float64`** | Represents continuous floating-point decimal numbers. |
| `datetime` (stdlib) | **`datetime64`** | Optimized format required for handling time-series operations. |

### 2. Statistical Distribution Profiling

A statistical summary allows data scientists to understand the spread and center of data distributions. It surfaces mathematical red flags such as large standard deviations (extreme data spread) or severe disparities between the median and maximum values, which point to the presence of heavy outliers.

---

## The Python Execution

### 1. Inspecting Data Types and Structure

To see how Pandas has classified your columns, you query structural attributes and summary methods.

```python
import pandas as pd

# Load sample dataset
file_path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/automobileDATA.csv"
df = pd.read_csv(file_path, header=None)

# 1. Check data types across all columns
# Returns a Series showing the data type assigned to each column index/name
print("--- DataFrame Data Types ---")
print(df.dtypes)

# 2. Check structural overview, missing entries, and memory footprint
print("\n--- DataFrame Structural Summary ---")
df.info()

```

### 2. Generating Statistical Summaries

Pandas provides a unified summary method via `.describe()`. Its behavior changes significantly depending on the arguments passed to it.

```python
# 1. Default Summary: Evaluates only numerical columns (int64, float64)
# Automatically skips categorical (object) text columns
print("--- Numerical Statistical Summary ---")
print(df.describe())

# 2. Comprehensive Summary: Evaluates ALL columns including object attributes
# Introduces categorical specific metrics (unique, top, freq)
print("\n--- Complete Column Summary (All Data Types) ---")
print(df.describe(include='all'))

```

---

## Interview Cheat Sheet

### 💡 The "Must-Knows" (Technical Gotchas)

* **The Object Type Catch-All:** In Pandas, if a single entry in a numeric column contains a non-numeric character (e.g., a currency symbol `$`, a comma `,`, a string like `"?"`, or a stray empty space), Pandas will automatically force the data type of that entire column to **`object`**. Identifying an expected numeric column (like price) showing up as an `object` type via `.dtypes` is a textbook data cleaning indicator.
* **Understanding `NaN` in Summary Tables:** When using `df.describe(include='all')`, you will observe several `NaN` (Not a Number) fields. This is normal behavior. Numerical metrics (like `mean`, `std`, `min`, `max`) cannot be computed for text columns, and text-specific metrics (like `unique`, `top`, `freq`) are not computed for numeric columns by default in that combined view.
* **Clarifying `.info()` Performance:** While displaying a large raw DataFrame textually truncation rules apply, running `df.info()` does not display individual data rows. Instead, it explicitly outputs data schema info: total row/index count, formal column labels, count of non-null elements present per column, and the live memory footprint occupied by the object.

### ❓ Common Interview Questions

#### Q1: If an expected numerical column (such as 'Price') returns a dtype of 'object', what does this indicate, and how does it restrict your analysis?

* **How to Answer:** This indicates data corruption or data quality anomalies within that column. If a column contains mostly numbers but has even one text character (such as a string placeholder `"?"` for a missing entry, a dollar sign, or a misplaced space), Pandas falls back to classifying the entire column as an `object` type. This restricts analysis because you cannot apply mathematical transformations, aggregations, or statistical modeling functions to that column without throwing a `TypeError`. You must clean the corrupt strings and cast it to a `float64` or `int64` first.

#### Q2: What is the difference between the statistics generated by `df.describe()` for a numerical column versus an object column?

* **How to Answer:** For numerical columns (`int64`, `float64`), `df.describe()` calculates descriptive spatial parameters: `count` (non-null rows), `mean` (average), `std` (standard deviation), `min`, `max`, and the quartile boundaries (`25%`, `50%` or median, and `75%`). For categorical/object columns, these mathematical properties are meaningless, so Pandas instead calculates:
* **`unique`**: The count of completely distinct string labels in that column.
* **`top`**: The modal element (the most frequently occurring string).
* **`freq`**: The absolute raw count of how many times that `top` modal element appears.



#### Q3: Why is `df.info()` considered one of the most critical methods to run immediately after importing a dataset?

* **How to Answer:** `df.info()` gives an immediate matrix diagnostic blueprint of your data. It allows you to rapidly check three vital structural points at once:
1. **Missing Data Diagnostics:** By comparing the total entries index count to the `Non-Null Count` listed for each column, you can pinpoint exactly which columns suffer from missing values.
2. **Data Type Verification:** It helps you spot type mismatches (e.g., numeric dimensions parsed as strings).
3. **Memory Management:** It prints the total RAM usage of the DataFrame. This allows you to evaluate whether you need to optimize your columns (such as downcasting floats or converting repetitive objects to the `category` dtype) before scaling your code.