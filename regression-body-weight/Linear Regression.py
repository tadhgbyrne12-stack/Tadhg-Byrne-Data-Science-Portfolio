import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import statsmodels.api as sm

# -------------------------
# 1) Load data
# -------------------------
# NOTE: Dataset path assumed relative to project directory
df = pd.read_csv("data/assessment_regression_dataset.csv")

# Target and predictors
y = df["weight"]
X = df.drop(columns=["weight", "age", "height"])

# -------------------------
# 2) Cross-validated Linear Regression
# -------------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)

rmse_scores = []
r2_scores = []

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))
    r2_scores.append(r2_score(y_test, preds))

print(f"Average RMSE: {np.mean(rmse_scores):.2f}")
print(f"Average R²: {np.mean(r2_scores):.3f}")

# -------------------------
# 3) OLS Regression for interpretability
# -------------------------
X_ols = sm.add_constant(X)
ols_model = sm.OLS(y, X_ols).fit()

print(ols_model.summary())

# -------------------------
# 4) Predicted vs Actual plot
# -------------------------
lr = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X)

plt.figure(figsize=(8, 8))
plt.scatter(y, y_pred, alpha=0.6, edgecolor="black")
plt.plot([y.min(), y.max()], [y.min(), y.max()])
plt.xlabel("Actual Weight (kg)")
plt.ylabel("Predicted Weight (kg)")
plt.title("Predicted vs Actual Body Weight from Skeletal Measurements")
plt.tight_layout()
plt.show()
