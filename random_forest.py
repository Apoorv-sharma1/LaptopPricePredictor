import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load and prepare data
df = pd.read_csv("refined_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Price'])
y = df['Price']

# Feature names for importance analysis
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Note: Random Forest does NOT require StandardScaler because it is tree-based
# (trees split on thresholds, not distances, so scale doesn't matter)

# Hyperparameter Tuning with GridSearchCV
print("Tuning Random Forest hyperparameters (this may take a moment)...")
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    RandomForestRegressor(random_state=42),
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

print("\n--- Random Forest Results ---")
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
print(f"Reason: Across all {model.n_estimators} trees, this feature was used most frequently at the top-level splits,")
print(f"        meaning it provides the most consistent reduction in prediction error.")
print(f"\nLeast Important Feature: {least_important['Feature']} (Score: {least_important['Importance']:.6f})")
print(f"Reason: This feature was almost never used for splitting across the forest,")
print(f"        meaning it provides almost no useful signal for predicting the price.")

# Visualizations
# 1. Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='darkorange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Random Forest: Actual vs Predicted (n_estimators={model.n_estimators})")
plt.show()

# 2. Top 10 Feature Importance Bar Chart
plt.figure(figsize=(12, 6))
top_10 = importance_df.head(10)
colors = plt.cm.Oranges(np.linspace(0.4, 1.0, 10))
plt.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color=colors)
plt.xlabel("Feature Importance Score")
plt.title("Top 10 Most Influential Features (Random Forest)")
plt.tight_layout()
plt.show()

# 3. Error Distribution
plt.figure(figsize=(10, 6))
errors = y_test - y_pred
plt.hist(errors, bins=30, edgecolor='black', color='darkorange', alpha=0.8)
plt.title("Error Distribution (Random Forest)")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error Line')
plt.legend()
plt.show()
