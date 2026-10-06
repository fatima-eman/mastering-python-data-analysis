## Module 3: Exploratory Data Analysis — Descriptive Statistics

Before you spend time building complicated machine learning models, it is essential to first explore your data. Exploratory Data Analysis (EDA) helps you describe the basic features of your dataset, get a quick summary of your samples, and catch potential errors early.

---

## Core Concept & "Why it Matters"

Descriptive statistics allow you to understand the shape, center, and spread of your data. We typically break this down into four main techniques:

### 1. Summarizing Numbers (`describe`)
The easiest way to understand your numeric data is by using the `.describe()` function. It automatically calculates core statistics for all numerical columns. Importantly, it **automatically skips any missing (`NaN`) values** so they do not break the math.

**What it finds:** Total count, average (mean), standard deviation (spread), minimum/maximum values, and quartiles.

### 2. Counting Categories (`value_counts`)
Categorical variables are discrete text labels (like a car's drive system: *front-wheel drive*, *rear-wheel drive*, etc.). You cannot calculate an average on text, so instead, we use `.value_counts()` to count exactly how many times each category appears. This helps spot dominant groups or imbalances.

### 3. Visualizing Distributions (Box Plots)
Box plots are an excellent way to visualize the spread of numeric data and compare it across different groups. They make it incredibly easy to spot **outliers** (extreme anomalies).

* **The Median:** The middle data point (50th percentile).
* **The Interquartile Range (IQR):** The middle 50% of your data (between the 25th and 75th percentiles).
* **The Extremes (Whiskers):** Calculated boundaries set at $1.5 \times \text{IQR}$ above the 75th percentile and below the 25th percentile.
* **Outliers:** Any individual dots plotted outside these extreme boundaries.

### 4. Finding Relationships (Scatter Plots)
When you have two continuous numerical variables (like *engine size* and *price*), you use a scatter plot to see if they are connected.

* **Predictor Variable (X-Axis):** The variable you are using to make a guess (e.g., Engine Size). It always goes on the horizontal axis.
* **Target Variable (Y-Axis):** The final outcome you are trying to predict (e.g., Price). It always goes on the vertical axis.
* *Outcome:* If the dots form an upward slope, it indicates a **positive linear relationship** (as engine size goes up, price goes up).

---

## The Python Execution

Here is how you execute these four core exploratory techniques in Python.

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Summarize all numerical columns (skips NaNs automatically)
df.describe()

# 2. Count the occurrences of categorical text values
drive_counts = df['drive-wheels'].value_counts().to_frame()
drive_counts.rename(columns={'drive-wheels': 'value_counts'}, inplace=True)
print(drive_counts)

# 3. Create a Box Plot to compare price across different drive-wheels
sns.boxplot(x='drive-wheels', y='price', data=df)
plt.title('Car Price by Drive Wheel Type')
plt.show()

# 4. Create a Scatter Plot to see if engine size predicts price
plt.scatter(x=df['engine-size'], y=df['price'])
plt.title('Engine Size vs. Car Price')
plt.xlabel('Engine Size (Predictor)')
plt.ylabel('Price (Target)')
plt.show()

```

---

## Interview Cheat Sheet

#### 💡 The "Must-Knows" (Technical Gotchas)

* **The `NaN` Blind Spot:** Pandas `.describe()` skips missing values quietly. Always check the `count` row against your total dataset length. If you have 1,000 rows but the count says 100, your statistics are biased by missing data.
* **Axis Reversal Mistake:** In live coding interviews, candidates sometimes flip the axes on scatter plots. Always remember: the independent variable (what you control or use to predict) is **X**, and the dependent variable (the final target/outcome) is **Y**.
* **Outlier Math Formula:** You are highly likely to be asked how a box plot defines an outlier. Memorize the multiplier: An outlier is any point beyond $1.5 \times \text{IQR}$ (Interquartile Range).

#### ❓ Common Interview Questions

**Category 1: Pandas Code Behavior**

* **Q1: How does the `.describe()` function handle missing data, and what is the danger of this?**
* **How to Answer:** Pandas automatically skips `NaN` values when calculating statistics. The danger is that if a column is missing 80% of its data, `.describe()` will still give you a perfectly normal-looking mean and standard deviation based on the remaining 20%. You must always check the `count` metric to ensure you have a valid sample size.


* **Q2: By default, `.describe()` ignores text columns. How do you force it to summarize categorical data, and what does it output?**
* **How to Answer:** You pass the argument `include='all'` or `include='object'`. Instead of math stats, it returns the `count` of non-null items, the number of `unique` categories, the `top` (most frequent) category, and its `freq` (frequency count).


* **Q3: How can you quickly turn `.value_counts()` from raw numbers into percentages?**
* **How to Answer:** By passing `normalize=True` into the function (e.g., `df['col'].value_counts(normalize=True)`). This transforms the raw counts into relative frequencies (decimals between 0 and 1), making it much easier to understand the proportion of each category.



**Category 2: Statistical Theory & Visualization**

* **Q4: What is the exact mathematical formula a box plot uses to identify an outlier?**
* **How to Answer:** A box plot finds outliers using the Interquartile Range ($\text{IQR} = Q3 - Q1$). It sets an upper extreme limit at $Q3 + (1.5 \times \text{IQR})$ and a lower extreme limit at $Q1 - (1.5 \times \text{IQR})$. Any data point that falls outside of these specific limits is plotted as an outlier dot.


* **Q5: Why do we often use the median instead of the mean to describe things like house prices or car prices?**
* **How to Answer:** The mean (average) is highly sensitive to extreme outliers. A few multi-million dollar luxury cars will pull the average artificially high. The median (the exact middle number) ignores these extremes, providing a much more realistic picture of the "typical" price.


* **Q6: What is the strict rule for placing variables on a scatter plot's axes?**
* **How to Answer:** The independent variable (the Predictor) must always go on the horizontal X-axis. The dependent variable (the Target you are trying to predict or explain) must always go on the vertical Y-axis.


* **Q7: If a scatter plot shows a strong positive trend, does that prove the X variable *causes* the Y variable to increase?**
* **How to Answer:** No. Correlation does not imply causation. A positive trend proves the two variables move together in a predictable way, making it a great feature for a machine learning model. However, it does not mathematically prove that X forces Y to happen (there could be a hidden third variable affecting both).



**Category 3: Business Scenarios & Data Imbalance**

* **Q8: Why is Exploratory Data Analysis (EDA) an absolute requirement before training a machine learning model?**
* **How to Answer:** Machine learning models blindly learn from the data you feed them. If you skip EDA, you might train a model on data full of massive outliers, highly skewed distributions, or unhandled missing values. EDA acts as a diagnostic health check to ensure your data is clean, balanced, and logically sound before you invest time in modeling.


* **Q9: Your `.value_counts()` shows 118 front-wheel drive cars and only 8 four-wheel drive cars. If you are building an AI to classify cars, why is this a major problem?**
* **How to Answer:** This is a severe class imbalance. Because four-wheel drive cars make up such a tiny fraction of the data, the AI will likely ignore them completely. It can achieve a high overall accuracy score by simply guessing "front-wheel drive" every single time, meaning it completely fails to learn how to identify the minority class.
