
# 📡 Telco Customer Churn Intelligence
> **Business Goal:** A predictive analytics dashboard designed to help telecom retention teams identify high-risk customers and understand the key drivers behind churn.

## 🚀 Live Demo
https://geekme20-github-github-io.onrender.com

## 🛠️ The Solution
I built an end-to-end machine learning pipeline that compares three different models (Logistic Regression, Random Forest, and XGBoost) to find the most accurate predictor. The final product is a **Dash application** that provides real-time risk scores.

### Key Features:
* **Predictive Scoring:** Instant churn probability based on customer profiles.
* **Model Explainability:** Visualizes the Top 10 features influencing the model's decision using SHAP-style importance.
* **Interactive What-If Analysis:** Adjust sliders for tenure and monthly charges to see how risk changes.

## 📊 Data & Insights
* **Top Driver:** Customers on 'Month-to-Month' contracts were 3x more likely to churn. However, lost some of the features
  during model deployment because of Dummy trap avoidance. Working on a solution to counter this!!
* **Model Performance:** The XGBoost model achieved an AUC-ROC of 0.8347
------------------------------------------------------------
BEST MODEL BY METRIC
------------------------------------------------------------
Accuracy        → Logistic Regression  (0.7991)
Precision       → Logistic Regression  (0.6454)
Recall          → XGBoost              (0.7487)
F1-Score        → Random Forest        (0.6269)
ROC-AUC         → Random Forest        (0.8404)
