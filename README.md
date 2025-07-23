# 📉 Customer Churn Prediction - Telecom

A machine learning project to predict whether a customer will churn (leave the service) or stay, based on their usage and contract data.

## 🚀 Overview

This project uses classification models to predict customer churn in a telecom company. It includes data preprocessing, exploratory analysis, feature encoding, scaling, model training, evaluation, and optimization using hyperparameter tuning.

---

## 📂 Dataset

- **Source**: Kaggle Telecom Churn Dataset
- **Target Column**: `Churn` (Yes/No)

---

## 🔧 Workflow

### 1. Data Preprocessing
- Handled missing and invalid values
- Converted `TotalCharges` to numeric
- Cleaned whitespaces and dropped unnecessary columns

### 2. Exploratory Data Analysis (EDA)
- Visualized churn rates across different features like contract type, internet service, payment method
- Compared tenure and charges for churn vs. non-churn

### 3. Feature Engineering
- Converted categorical features using `LabelEncoder` / `pd.get_dummies`
- Scaled numeric columns (`tenure`, `MonthlyCharges`, `TotalCharges`)

### 4. Modeling
- Train-test split (80/20)
- Trained the following classifiers:
  - Decision Tree ✅
  - Logistic Regression
  - Random Forest
  - Gradient Boosting

### 5. Evaluation
- Accuracy, precision, recall, F1-score
- Confusion matrix
- Observed trade-offs: Some models had better recall for churn class

### 6. Hyperparameter Tuning
- Used `GridSearchCV` on Decision Tree to improve recall for churn (class 1)
- Best model:  
  ```python
  {'criterion': 'entropy', 'max_depth': 3, 'min_samples_leaf': 1, 'min_samples_split': 2}
