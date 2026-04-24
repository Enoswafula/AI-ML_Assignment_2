# Titanic dataset exploration

## Overview

This project explores the Titanic dataset and builds a data preprocessing pipeline to prepare the data for machine learning models that predict passenger survival.

The focus is on:

* Data cleaning
* Exploratory Data Analysis (EDA)
* Feature engineering
* Data transformation

---

## Dataset

* Source: Kaggle Titanic Dataset
* Files used:

  * `train.csv`
  * `test.csv`

---

## Exploratory Data Analysis (EDA)

Initial analysis was performed to understand the dataset structure and distributions.

Key steps:

* Checked dataset structure using `info()`
* Generated statistical summaries using `describe()`
* Visualized distributions of features like **Age** and **Fare**
* Identified missing values across columns

---

## Data Cleaning

### Handling Missing Values

* Replaced empty values with `NaN`
* Dropped **Cabin** column due to excessive missing values
* Filled missing values:

  * `Age` → median
  * `Embarked` → most frequent value

### Duplicate Handling

* Checked for duplicate rows
* No significant duplicates found

### Dropping Irrelevant Features

* Removed columns like:

  * `Name` (not useful for modeling)

---

## Feature Engineering

### Categorical Encoding

Converted categorical variables into numeric form using one-hot encoding:

* `Sex`
* `Embarked`
* `Title`

```python
pd.get_dummies(..., drop_first=True)
```

---

### Ordinal Encoding

Created ordered categories for age groups:

```python
age_mapping = {
    'Child': 0,
    'Teen': 1,
    'Adult': 2,
    'Senior': 3
}
```

---

## Feature Transformation

### Handling Skewness

* Observed skewness in **Fare**
* Applied log transformation:

```python
np.log1p(Fare)
```

This helped normalize the distribution and reduce outliers.

---

### Feature Scaling

Used standardization to normalize numerical features:

```python
from sklearn.preprocessing import StandardScaler
```

Scaled features:

* Age
* Fare
* FamilySize
* Pclass

---

## Tools & Libraries Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## Project Workflow

1. Load dataset
2. Perform EDA
3. Clean data
4. Handle missing values
5. Encode categorical features
6. Transform skewed features
7. Scale numerical features

---

## Key Insights

* Missing data is a major issue in the dataset (especially Cabin)
* Fare distribution is highly skewed and requires transformation
* Feature engineering significantly improves data quality

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

```python
# Run the notebook
jupyter notebook titanic.ipynb
```
---

## Future Improvements

* Train and evaluate multiple machine learning models
* Perform hyperparameter tuning
* Deploy model as an API or web app

---
## 📌 Author
**Enos Wafula**  
Aspiring Data Scientist | Python | SQL | Power BI
