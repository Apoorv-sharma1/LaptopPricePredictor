import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Load and prepare data
df = pd.read_csv("refined_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

# XGBoost requires clean column names (no special characters)
df.columns = df.columns.str.replace(r'[\[\]<>]', '_', regex=True)

X = df.drop(columns=['Price'])
y = df['Price']

# Feature names for importance analysis
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Note: XGBoost is also tree-based, so StandardScaler is NOT required.

# Hyperparameter Tuning with GridSearchCV
print("Tuning XGBoost hyperparameters (this may take a moment)...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    XGBRegressor(random_state=42, verbosity=0),
    param_grid,
    cv=5,
    scoring='r2',
    n_jobs=-1  # Use all CPU cores for speed
)
grid_search.fit(X_train, y_train)

print(f"Best Parameters found: {grid_search.best_params_}")

model = grid_search.best_estimator_
y_pred = model.predict(X_test)

# Evaluation (Convert back from log scale for real-world metrics)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
accuracy = 100 - (np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100)

print("\n--- XGBoost Results ---")
print("R2 Score (on log scale):", r2)
print("MAE (Real Price):", mae)
print("RMSE (Real Price):", rmse)
print("Accuracy:", accuracy)

if r2 > 0.8:
    print("Model Performance: Very Good")
elif r2 > 0.6:
    print("Model Performance: Good")
elif r2 > 0.4:
    print("Model Performance: Average")
else:
    print("Model Performance: Poor")

# Feature Importance Analysis
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

most_important = importance_df.iloc[0]
least_important = importance_df.iloc[-1]

print("\n--- Feature Importance Analysis ---")
print(f"Most Important Feature : {most_important['Feature']} (Score: {most_important['Importance']:.4f})")
print(f"Reason: XGBoost uses Gradient Boosting, where each tree corrects the errors of the previous one.")
print(f"        This feature consistently caused the biggest error reductions across all boosting rounds.")
print(f"\nLeast Important Feature: {least_important['Feature']} (Score: {least_important['Importance']:.6f})")
print(f"Reason: XGBoost found this feature contributed almost nothing to reducing prediction error")
print(f"        across all boosting iterations, so it was essentially ignored by the model.")

# Visualizations
# 1. Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
best_lr = grid_search.best_params_['learning_rate']
best_depth = grid_search.best_params_['max_depth']
plt.title(f"XGBoost: Actual vs Predicted (lr={best_lr}, depth={best_depth})")
plt.show()

# 2. Top 10 Feature Importance Bar Chart
plt.figure(figsize=(12, 6))
top_10 = importance_df.head(10)
colors = plt.cm.cool(np.linspace(0.3, 0.9, 10))
plt.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color=colors)
plt.xlabel("Feature Importance Score")
plt.title("Top 10 Most Influential Features (XGBoost)")
plt.tight_layout()
plt.show()

# 3. Error Distribution
plt.figure(figsize=(10, 6))
errors = y_test - y_pred
plt.hist(errors, bins=30, edgecolor='black', color='teal', alpha=0.8)
plt.title("Error Distribution (XGBoost)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error Line')
plt.legend()
plt.show()
