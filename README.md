# 💰 Salary Prediction — Simple Linear Regression

Built a simple linear regression model to predict salary based on years of experience, using Scikit-learn.

## 📌 Overview
Demonstrates the full ML workflow — data preparation, train-test split, model training, evaluation, and residual analysis — using a synthetic salary dataset.

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## 🔍 Steps Followed
1. **Data Preparation** — Generated a realistic synthetic dataset (100 samples) modeling salary as a function of experience with added noise.
2. **Data Splitting** — 80% training, 20% testing (`train_test_split`).
3. **Model Building** — Trained a `LinearRegression()` model from Scikit-learn.
4. **Model Evaluation** — Measured performance using MSE, RMSE, and R² Score.
5. **Residual Analysis** — Checked prediction errors for bias/patterns.

## 📊 Results
- **Model Equation:** Salary = 7786 × Experience + 31143
- **R² Score:** 0.9686 (explains 96.9% of variance)
- **RMSE:** ₹6,468

## 📁 Files
- `linear_regression_salary.py` — full comment done Python 
- `*.png` — regression line, train/test split, residual plots
