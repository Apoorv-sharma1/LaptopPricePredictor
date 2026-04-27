import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load and prepare data
df = pd.read_csv("refined_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Price'])
y = df['Price']

# Feature names for importance analysis
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

# Scaling features (Helps with feature importance interpretation)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

# Evaluation (Convert back from log scale for real-world metrics)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
accuracy = 100 - (np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100)

print("\n--- Linear Regression Results ---")
print("R2 Score (on log scale):", r2)
print("MAE (Real Price):", mae)
print("RMSE (Real Price):", rmse)
print("Accuracy:", accuracy)

# Feature Importance Analysis
coeffs = model.coef_
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': np.abs(coeffs)})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

most_important = importance_df.iloc[0]
least_important = importance_df.iloc[-1]

print("\n--- Feature Importance Analysis ---")
print(f"Most Important Feature: {most_important['Feature']} (Score: {most_important['Importance']:.2f})")
print(f"Reason: Since we scaled the features, the coefficient magnitude shows that this feature has the strongest weight in the linear equation.")
print(f"\nLeast Important Feature: {least_important['Feature']} (Score: {least_important['Importance']:.2f})")
print(f"Reason: This feature has the smallest coefficient, meaning it adds the least amount of value to the final price prediction.")

# Visualizations
# 1. Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Linear Regression: Actual vs Predicted")
plt.show()

# 2. Top 10 Feature Importance
plt.figure(figsize=(12, 6))
top_10 = importance_df.head(10)
plt.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color='dodgerblue')
plt.xlabel("Importance (Absolute Coefficient)")
plt.title("Top 10 Most Influential Features (Linear Regression)")
plt.show()

# 3. Error Distribution
plt.figure(figsize=(10, 6))
errors = y_test - y_pred
plt.hist(errors, bins=30, edgecolor='black', color='skyblue')
plt.title("Error Distribution (Linear Regression)")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()
