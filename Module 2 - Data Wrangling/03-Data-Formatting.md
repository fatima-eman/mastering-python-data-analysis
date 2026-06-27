## Module 2: Data Wrangling — Data Formatting in Python

Data formatting means bringing your data into a clear, common standard. When data is collected from different sources or by different people, it often arrives in a messy mix of formats, units, and conventions. Formatting fixes this so you can make fair, apple-to-apple comparisons across your entire dataset.

---

## Core Concept & "Why it Matters"

If your data is not formatted correctly, your data analysis will be inaccurate. There are three main areas where data formatting is necessary:

### 1. Consistent Labels (Text Formatting)

People often write the same piece of information in different ways. For example, New York City might appear in a dataset as `NY`, `N.Y.`, `ny`, or `New York`.

* *When variance is good:* Sometimes, keeping these variations is helpful—such as when looking for spelling patterns to detect fraud.
* *When consistency is needed:* Most of the time, you want to merge them into a single format (like `New York`) so your statistical models can count and analyze them as the exact same thing.

### 2. Standard Units (Unit Conversion)

Datasets often use regional units that may not match what you or your company need. For example, a used car dataset might measure fuel efficiency in **miles per gallon (mpg)**, but your team needs it in **liters per 100 kilometers (L/100km)**. Standardizing your units keeps your reports consistent.

### 3. Correct Data Types

When you load data into Python, Pandas tries its best to guess the data types, but it often gets them wrong. A common error is when a column of numbers (like `price`) is imported as text (known as an **object** type in Pandas).

If you leave numbers stored as text:

* You cannot perform math operations on them (like finding the average price).
* Data models will break or behave strangely.
* Valid numbers might accidentally be treated as missing data (`NaN`).

---

## The Python Execution

### 1. Changing Units and Renaming Columns

To convert units, you can apply a mathematical formula to an entire column at once. After changing the values, you should rename the column so the label accurately reflects the new unit.

```python
import pandas as pd

# Create a sample dataset with miles per gallon (mpg)
data = {'car': ['Sedan A', 'SUV B', 'Truck C'], 'city-mpg': [22, 18, 15]}
df = pd.DataFrame(data)

# 1. Convert MPG to Liters per 100 Kilometers (Formula: 235 / mpg)
df['city-mpg'] = 235 / df['city-mpg']

# 2. Rename the column to reflect the new unit
# We pass a dictionary to the 'columns' argument: {'old_name': 'new_name'}
df.rename(columns={'city-mpg': 'city-L/100km'}, inplace=True)

print("DataFrame after unit conversion and column rename:")
print(df)

```

### 2. Checking and Converting Data Types

You can check data types using `.dtypes`. If a column has the wrong type, fix it using the `.astype()` method.

```python
# Create a sample dataset where numbers are accidentally trapped as text strings
data = {'car': ['Sedan A', 'SUV B', 'Truck C'], 'price': ['12000', '24000', '35000']}
df_cars = pd.DataFrame(data)

# Check the initial data types
print("Initial Data Types:")
print(df_cars.dtypes) 
# Note: 'price' will show up as 'object' (which means text/string in Pandas)

# Convert the 'price' column from text (object) to integers (int)
df_cars['price'] = df_cars['price'].astype('int')

print("\nData Types after conversion:")
print(df_cars.dtypes)
# Note: 'price' is now 'int64', meaning you can now calculate averages or sums!

```

---

## Interview Cheat Sheet

### 💡 The "Must-Knows" (Technical Gotchas)

* **`dtypes` vs `dtypes()`:** A very common beginner mistake in interviews is writing `df.dtypes()`. Remember that `dtypes` is an *attribute* (a property) of the DataFrame, not a function. It does **not** take parentheses.
* **`astype()` does not change data in place:** Just running `df['price'].astype('int')` will look like it worked, but it won't actually update your DataFrame. It creates a temporary copy. You must assign it back to the column: `df['price'] = df['price'].astype('int')`.
* **The `astype()` Crash Hazard:** If you try to run `.astype('int')` on a column that contains even a single text character (like a dollar sign `$12,000` or a letter `Unknown`), Python will throw a major error and crash. You must strip away special characters before converting data types.

### ❓ Common Interview Questions

#### Q1: What does data formatting mean, and why is it an important step in data cleaning?

* **How to Answer:** Data formatting means converting data into a standard, uniform format so it can be compared accurately. It is important because raw data often comes from different sources using different units, text abbreviations, or data types. Proper formatting ensures consistency, allows you to perform correct mathematical calculations, and prevents errors when building machine learning models later.

#### Q2: If a column containing car prices is loaded into Pandas as an `object` data type, why is this a problem and how do you fix it?

* **How to Answer:** This is a problem because `object` means Pandas is treating the numbers as text strings. As a result, you won't be able to perform any math operations on the column, such as finding the average or maximum price, and analytical tools will treat them incorrectly. To fix this, you use the `.astype()` method to convert the column to a numerical type, like `int` or `float`, and assign it back to the column: `df['price'] = df['price'].astype('int')`.

#### Q3: How do you safely rename a column in Pandas after changing its measurement unit?

* **How to Answer:** You use the `.rename()` method on the DataFrame. You pass a dictionary to the `columns` parameter, where the key is the old column name and the value is the new column name (for example, `columns={'city-mpg': 'city-L/100km'}`). To make sure this change saves directly to the existing DataFrame without creating a new copy, you should include the argument `inplace=True`.