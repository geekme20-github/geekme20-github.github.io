# 📡 Telco Customer Churn Intelligence
> **Business Goal:** A predictive analytics dashboard designed to help telecom retention teams identify high-risk customers and understand the key drivers behind churn.

## 🚀 Live Demo
[Link to your Render App (once deployed)]

## 🛠️ The Solution
I built an end-to-end machine learning pipeline that compares three different models (Logistic Regression, Random Forest, and XGBoost) to find the most accurate predictor. The final product is a **Dash application** that provides real-time risk scores.

### Key Features:
* **Predictive Scoring:** Instant churn probability based on customer profiles.
* **Model Explainability:** Visualizes the Top 10 features influencing the model's decision using SHAP-style importance.
* **Interactive What-If Analysis:** Adjust sliders for tenure and monthly charges to see how risk changes.

## 📊 Data & Insights
* **Top Driver:** Customers on 'Month-to-Month' contracts were 3x more likely to churn.
* **Model Performance:** The XGBoost model achieved an AUC-ROC of [Your Score, e.g., 0.84].
