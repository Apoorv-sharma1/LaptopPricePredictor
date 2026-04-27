import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# Load and prepare data
df = pd.read_csv("refined_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop("Price", axis=1)
y = df["Price"]

# Feature names for importance analysis
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Using GridSearchCV to find the best hyperparameters
print("Tuning Decision Tree hyperparameters...")
param_grid = {
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=5, scoring='r2')
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

print("\n--- Decision Tree Results ---")
print("R2 Score (on log scale):", r2)
print("MAE (Real Price):", mae)
print("RMSE (Real Price):", rmse)
print("Accuracy:", accuracy)

# Feature Importance Analysis
importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

most_important = importance_df.iloc[0]
least_important = importance_df.iloc[-1]

print("\n--- Feature Importance Analysis ---")
print(f"Most Important Feature: {most_important['Feature']} (Score: {most_important['Importance']:.4f})")
print(f"Reason: This feature provided the highest reduction in impurity during the tree splits, making it the primary factor in determining the price.")
print(f"\nLeast Important Feature: {least_important['Feature']} (Score: {least_important['Importance']:.4f})")
print(f"Reason: This feature was rarely or never used for splitting, indicating it has little to no predictive power in this model.")

# Visualizations
# 1. Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Decision Tree: Actual vs Predicted (Max Depth={grid_search.best_params_['max_depth']})")
plt.show()

# 2. Top 10 Feature Importance
plt.figure(figsize=(12, 6))
top_10 = importance_df.head(10)
plt.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color='violet')
plt.xlabel("Importance Score")
plt.title("Top 10 Most Influential Features (Decision Tree)")
plt.show()

# 3. Error Distribution
plt.figure(figsize=(10, 6))
errors = y_test - y_pred
plt.hist(errors, bins=30, edgecolor='black', color='mediumorchid')
plt.title("Error Distribution (Decision Tree)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()

