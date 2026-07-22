# Credit Card Default Prediction

This project focuses on predicting whether a customer will default on their credit card payment using demographic, financial, and credit-related information. The objective is to build an interpretable and reliable machine learning pipeline that can identify high-risk customers while addressing challenges such as missing values, feature engineering, and class imbalance.

---
## Dataset

The dataset contains information related to customers' demographics, employment, income, credit history, repayment behavior, and credit utilization. The target variable is **`credit_card_default`**, making this a binary classification problem.

- **Rows:** 45,528
- **Features:** 18 input features + 1 target variable
---

## Data Cleaning & Feature Engineering

- Handled missing values using row removal, mode imputation, and KNN imputation.
- Removed inconsistent and invalid records.
- Treated outliers in important numerical features.
- Applied log transformation to reduce skewness.
- Created new features such as **Years of Experience** and **Employment Status**.
- Performed feature relationship analysis using correlation and VIF.

---

## Model Pipeline

The complete workflow was implemented using **Scikit-learn Pipelines** and **ColumnTransformer** to ensure consistent preprocessing during training and evaluation.

**Preprocessing Techniques**
- Standard Scaling
- One-Hot Encoding
- Target Encoding
- SMOTENC (Class Imbalance Handling)

---

## Models Used

| Model | Hyperparameter Tuning |
|--------|-----------------------|
| Logistic Regression | GridSearchCV |
| Random Forest | RandomizedSearchCV |
| XGBoost | RandomizedSearchCV |

---

## Model Evaluation

The models were evaluated using multiple classification metrics:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve

---

## Model Explainability

To improve model interpretability, **SHAP (SHapley Additive exPlanations)** was used to understand feature contributions through:

- Feature Importance
- Summary Plot
- Dependence Plot
- Waterfall Plot

---