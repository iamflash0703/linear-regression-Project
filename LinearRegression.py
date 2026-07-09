"""
Project 2: Simple Linear Regression

Goal: Predict Salary based on Years of Experience using a simple
linear regression model, and evaluate how well it performs.
"""

# ---------- STEP 0: Import Libraries ----------
import pandas as pd                                  # data handling
import numpy as np                                    # numeric operations
import matplotlib.pyplot as plt                        # plotting
import seaborn as sns                                  # nicer plots
from sklearn.model_selection import train_test_split    # splitting data
from sklearn.linear_model import LinearRegression       # the ML model
from sklearn.metrics import mean_squared_error, r2_score  # evaluation metrics

sns.set_style("darkgrid")

# ---------- STEP 1: Data Preparation ----------
# No public "Salary vs Experience" CSV is directly downloadable in this
# environment, so we generate a realistic synthetic dataset instead
# (this is a common, accepted practice for this specific mini-project,
# as mentioned in the project brief itself).
np.random.seed(42)  # for reproducibility - same random numbers every run

n_samples = 100
# Years of experience: random values between 0 and 15 years
experience = np.random.uniform(0, 15, n_samples)

# Salary formula: base salary + (experience * increment) + random noise
# This mimics a real-world relationship where salary grows with experience
# but isn't perfectly linear (real data always has some noise/randomness)
base_salary = 30000
increment_per_year = 8000
noise = np.random.normal(0, 8000, n_samples)  # random noise (people vary!)
salary = base_salary + (experience * increment_per_year) + noise

# Put into a DataFrame (like a spreadsheet/table)
df = pd.DataFrame({
    "YearsExperience": experience,
    "Salary": salary
})

print("Dataset shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())

# ---------- STEP 2: Data Splitting ----------
# X = input feature (what we use to predict), y = target (what we predict)
X = df[["YearsExperience"]]   # double brackets -> keeps it as a DataFrame (2D)
y = df["Salary"]              # single bracket -> Series (1D)

# Split into 80% training data, 20% testing data
# Model learns from training data, and we check its performance on unseen test data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------- STEP 3: Model Building & Training ----------
model = LinearRegression()   # create the model object
model.fit(X_train, y_train)  # train it (finds the best-fit line)

# The line equation is: Salary = slope * Experience + intercept
slope = model.coef_[0]
intercept = model.intercept_
print(f"\nModel equation: Salary = {slope:.2f} * Experience + {intercept:.2f}")

# ---------- STEP 4: Predictions ----------
y_pred = model.predict(X_test)

# ---------- STEP 5: Model Evaluation ----------
mse = mean_squared_error(y_test, y_pred)     # average squared error
rmse = np.sqrt(mse)                          # root mean squared error (same unit as salary)
r2 = r2_score(y_test, y_pred)                # how much variance is explained (0 to 1)

print(f"\nMean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2 Score): {r2:.4f}")

# ---------- STEP 6: Visualization ----------

# 6a. Scatter plot with regression line (on test data)
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color="#4C9AFF", label="Actual Salary", s=60, alpha=0.8)
plt.plot(X_test, y_pred, color="#E74C3C", linewidth=2.5, label="Predicted (Regression Line)")
plt.title("Salary vs Experience — Regression Fit (Test Data)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.savefig("regression_line.png", dpi=150)
plt.close()

# 6b. Full dataset scatter (train + test) to see overall trend
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color="#2ECC71", label="Training Data", alpha=0.6)
plt.scatter(X_test, y_test, color="#4C9AFF", label="Testing Data", alpha=0.6)
all_X = np.linspace(0, 15, 100).reshape(-1, 1)
all_pred = model.predict(all_X)
plt.plot(all_X, all_pred, color="#E74C3C", linewidth=2, label="Regression Line")
plt.title("Full Dataset — Train/Test Split with Regression Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.tight_layout()
plt.savefig("full_dataset_fit.png", dpi=150)
plt.close()

# 6c. Residual plot (errors) - checks if model is unbiased
residuals = y_test - y_pred
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color="#F39C12", alpha=0.7)
plt.axhline(y=0, color="#E74C3C", linestyle="--", linewidth=2)
plt.title("Residual Plot (Prediction Errors)")
plt.xlabel("Predicted Salary")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.savefig("residual_plot.png", dpi=150)
plt.close()

print("\nAll plots saved successfully.")

# ---------- STEP 7: Key Insights ----------
print("\nKey Insights:")
print(f"1. For every additional year of experience, salary increases by "
      f"approximately {slope:.0f} rupees.")
print(f"2. The model explains {r2*100:.1f}% of the variance in salary "
      f"(R2 = {r2:.4f}) — a strong fit for a simple linear model.")
print(f"3. Average prediction error (RMSE) is {rmse:.0f} rupees.")