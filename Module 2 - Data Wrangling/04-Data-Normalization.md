## Module 2: Data Wrangling — Data Normalization in Python

Data normalization is a data setup step where we adjust the values in different columns so they all fit into a similar, consistent range.

---

## Core Concept & "Why it Matters"

When you look at a raw dataset, different features (columns) are often measured in completely different ranges. For example:

* In a car dataset: **Car Length** might range from 150 to 250, while **Car Width** ranges from 50 to 100.
* In a financial dataset: **Age** ranges from 0 to 100, while **Income** ranges from $20,000 to $500,000.

### Why is this a problem?

If you feed this raw data into a machine learning model (like Linear Regression), the model will look at the massive numbers in the **Income** column and automatically assume Income is much more important than Age. The model gets biased purely because one number is physically larger than the other, not because it is actually a better predictor.

### What does Normalization do?

Normalization levels the playing field. It shrinks or shifts the numbers so that every feature has the exact same scale and range. This ensures:

1. **Fair Comparison:** Every feature gets an equal chance to influence the final model.
2. **Speed:** It helps computers run the complex math behind data models much faster.

---

## The Three Normalization Methods

The transcript highlights three popular ways to scale down your data:

### 1. Simple Feature Scaling

* **The Math:** Divide every individual value by the maximum value in that column ($x_{new} = \frac{x}{x_{max}}$).
* **The Result:** All your new values will land neatly between **0 and 1**.

### 2. Min-Max Scaling

* **The Math:** Take your value, subtract the lowest value in the column, and then divide by the total range (maximum minus minimum).
* **The Result:** This strictly forces all values to range between **0 and 1**. The absolute lowest original value becomes exactly `0`, and the absolute highest becomes exactly `1`.

### 3. Z-Score Normalization (Standardization)

* **The Math:** For each value, subtract the average (mean) of the column, and then divide by the standard deviation.
* **The Result:** The new values will sit around **0**. Most of your data will fall between **-3 and +3**. If a value is negative, it means it is below average; if it is positive, it is above average.

---

## The Python Execution

Here is how you apply all three normalization techniques to a column (like a car's `length`) using Pandas:

```python
import pandas as pd

# Let's create a sample dataset of car lengths
data = {'car': ['Car A', 'Car B', 'Car C', 'Car D'],
        'length': [150.0, 175.0, 210.0, 250.0]}
df = pd.DataFrame(data)

# Make copies of the data frame to show each method clearly
df_simple = df.copy()
df_minmax = df.copy()
df_zscore = df.copy()

# ---------------------------------------------------------
# Method 1: Simple Feature Scaling (Divide by Max)
# ---------------------------------------------------------
df_simple['length'] = df_simple['length'] / df_simple['length'].max()
print("Simple Feature Scaling Result (0 to 1):")
print(df_simple[['car', 'length']])


# ---------------------------------------------------------
# Method 2: Min-Max Scaling
# ---------------------------------------------------------
# Formula: (X - min) / (max - min)
minimum = df_minmax['length'].min()
maximum = df_minmax['length'].max()

df_minmax['length'] = (df_minmax['length'] - minimum) / (maximum - minimum)
print("\nMin-Max Scaling Result (Strictly 0 to 1):")
print(df_minmax[['car', 'length']])


# ---------------------------------------------------------
# Method 3: Z-Score Normalization
# ---------------------------------------------------------
# Formula: (X - mean) / standard_deviation
mean_value = df_zscore['length'].mean()
std_value = df_zscore['length'].std()

df_zscore['length'] = (df_zscore['length'] - mean_value) / std_value
print("\nZ-Score Result (Centered around 0, usually -3 to +3):")
print(df_zscore[['car', 'length']])

```

---

## Interview Cheat Sheet

### 💡 The "Must-Knows" (Technical Gotchas)

* **Outlier Sensitivity:** **Min-Max Scaling** is highly sensitive to extreme outliers. If you have one car with an accidental length of `9999`, it will ruin the scaling for all other normal cars, crushing them down close to `0`. **Z-Score Normalization** handles extreme outliers much better because it uses the average and spread of the data.
* **`mean` and `std` are methods:** In Pandas, `.mean()` and `.std()` are functions that compute values. Do not forget to include the parentheses `()` when calling them on a column, or your code will fail to calculate the numbers.

### ❓ Common Interview Questions

#### Q1: Why do we normalize data before feeding it into algorithms like Linear Regression?

* **How to Answer:** We normalize data because features often use vastly different scales (like age vs. income). If left unscaled, features with larger numbers will naturally dominate the model's math, biasing the results. Normalization brings all columns to a consistent range so that every feature has a fair and equal impact on the model.

#### Q2: What is the main difference between Min-Max scaling and Z-Score normalization?

* **How to Answer:** The main difference is the boundary and how they handle outliers. **Min-Max scaling** bounds the data strictly between `0` and `1`, but can be easily ruined if there are extreme outlier numbers. **Z-Score normalization** does not have a strict boundary; it centers the data around a mean of `0` with a standard deviation of `1` (typically ranging from `-3` to `+3`), making it much safer to use when your data contains extreme outliers.

#### Q3: How do you write a Z-Score normalization from scratch using Pandas?

* **How to Answer:** You calculate the mean of the column using `.mean()` and the standard deviation using `.std()`. Then, you subtract the mean from the column and divide the result by the standard deviation. In code, it looks like this: `df['col'] = (df['col'] - df['col'].mean()) / df['col'].std()`.