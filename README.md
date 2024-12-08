# Insurance Fraud Detection: A Machine Learning Approach

## Overview

This project aims to predict insurance fraud using machine learning techniques. The dataset contains information about various insurance policies and claims, including details about the insured person, claims, and incident details. The primary goal is to identify whether a claim is fraudulent based on these features. The project uses various machine learning models, performs exploratory data analysis (EDA), and applies techniques like class balancing to improve the prediction of fraudulent claims.

---

## Table of Contents

- [Data Collection and Preprocessing](#data-collection-and-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
  - [Logistic Regression](#logistic-regression)
  - [Random Forest](#random-forest)
  - [Hyperparameter Tuning](#hyperparameter-tuning)
  - [Handling Class Imbalance](#handling-class-imbalance)
  - [Evaluation](#evaluation)
  - [Confusion Matrix and ROC Curve](#confusion-matrix-and-roc-curve)
- [Conclusion](#conclusion)

---

## Data Collection and Preprocessing

### Dataset
The dataset contains the following key features:

- **Policy Features**: `policy_number`, `policy_csl`, `policy_annual_premium`, `umbrella_limit`
- **Insured Information**: `insured_sex`, `insured_education_level`, `insured_occupation`, `insured_relationship`
- **Incident Information**: `incident_type`, `collision_type`, `incident_severity`, `incident_hour_of_the_day`, `property_damage`
- **Claims Information**: `injury_claim`, `property_claim`, `vehicle_claim`, `fraud_reported`

### Data Preprocessing

1. **Removing Unnecessary Columns**: The columns `policy_number`, `policy_bind_date`, `policy_state`, etc., are dropped since they were irrelevant to predicting fraud.
   
2. **Handling Missing Values**:
   - For categorical columns, missing values were imputed with the mode (most frequent value).
   - For numerical columns, missing values were imputed with the median.

3. **Feature Encoding**: Categorical variables were transformed into numerical representations using one-hot encoding to make them suitable for machine learning algorithms.

4. **Scaling**: Features were scaled using `StandardScaler` to bring all the values to a similar range for improved model performance.

---

## Exploratory Data Analysis (EDA)

The objective of the EDA was to understand the relationships between variables and the distribution of the target variable (`fraud_reported`).

Key EDA steps:
- **Univariate Analysis**: Visualizations like histograms, bar plots, and box plots were used to examine the distributions of features.
- **Bivariate Analysis**: Correlations between independent variables were evaluated, and plots like pairplots were used to visualize the relationship between features and the target.
- **Missing Data**: Identified and handled missing data using appropriate imputation methods.

**Visuals:**
1. Distribution of Fraud Reported
2. Pairplot of Features
3. Correlation Heatmap

---

## Feature Engineering

In this step, new features were created and redundant or highly correlated features were removed:

1. **Feature Encoding**: Categorical columns were one-hot encoded.
2. **Handling Multicollinearity**: Correlated numerical features were removed to avoid multicollinearity, ensuring the stability of the models.
3. **Binning**: Numerical features like `policy_annual_premium` were binned into discrete categories to reduce the impact of extreme values.

---

## Modeling

Several machine learning algorithms were applied to the dataset to predict insurance fraud.

### Logistic Regression

A simple yet powerful algorithm was used to establish a baseline model for predicting fraud. The model was evaluated using accuracy, precision, recall, and F1-score.

**Visuals:**
1. Model Performance Metrics (Logistic Regression)

### Random Forest

Random Forest was used to capture non-linear relationships in the data. This ensemble method can handle high-dimensional data and interactions between variables.

**Visuals:**
1. Random Forest Model Performance

### Hyperparameter Tuning with GridSearchCV

To optimize model performance, we tuned the hyperparameters of the Random Forest model using `GridSearchCV`.

**Visuals:**
1. Hyperparameter Tuning Results

### Handling Class Imbalance

The dataset suffers from class imbalance, where fraudulent claims (`fraud_reported`) are much less frequent than non-fraudulent claims. To mitigate this issue, we applied **SMOTE (Synthetic Minority Over-sampling Technique)** to oversample the minority class.

**Visuals:**
1. Class Distribution After SMOTE

### Evaluation

The model's performance was evaluated using various metrics:
- **Confusion Matrix**: To visually inspect the true positives, false positives, true negatives, and false negatives.
- **Classification Report**: To evaluate precision, recall, and F1-score.
- **ROC Curve**: To assess the trade-off between true positive rate and false positive rate.

**Visuals:**
1. Confusion Matrix
2. ROC Curve

---

## Confusion Matrix and ROC Curve

### Confusion Matrix Plot

A confusion matrix plot was generated to visualize the model's performance in terms of true positives, false positives, true negatives, and false negatives.

**Visuals:**
1. Confusion Matrix Plot

### ROC Curve

The ROC curve was plotted to evaluate the model’s performance at different classification thresholds.

**Visuals:**
1. ROC Curve

---

## Conclusion

This project demonstrates the process of detecting insurance fraud using machine learning techniques. By applying various algorithms, handling class imbalance, and fine-tuning hyperparameters, we were able to build a predictive model with a balanced evaluation. The use of visualization techniques like confusion matrices and ROC curves provides a deeper understanding of the model's performance.

Future improvements could include:
- Using more advanced models such as **XGBoost** or **LightGBM**.
- Further feature engineering, such as adding temporal features or aggregating claims data.
- Implementing cost-sensitive learning to penalize misclassifying fraud cases more heavily.


```