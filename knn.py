import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Load and prepare data
df = pd.read_csv("refined_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Price'])
y = df['Price']

# Store feature names for importance analysis
feature_names = X.columns

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Elbow Method to find the optimal k
rmse_values = []
k_range = range(1, 21)

print("Starting Elbow Method to find optimal k...")
for k in k_range:
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred_k = knn.predict(X_test)
    rmse_k = np.sqrt(mean_squared_error(y_test, y_pred_k))
    rmse_values.append(rmse_k)
    print(f"k={k}, RMSE={rmse_k:.4f}")

# Selecting the best k (k with minimum RMSE)
optimal_k = k_range[np.argmin(rmse_values)]
print(f"\nOptimal k selected: {optimal_k}")

# Training the final model with the optimal k
model = KNeighborsRegressor(n_neighbors=optimal_k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Evaluation (Convert back from log scale for real-world metrics)
y_test_real = np.expm1(y_test)
y_pred_real = np.expm1(y_pred)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test_real, y_pred_real)
mse = mean_squared_error(y_test_real, y_pred_real)
rmse = np.sqrt(mse)
accuracy = 100 - (np.mean(np.abs((y_test_real - y_pred_real) / y_test_real)) * 100)

print("\n--- Final Model Evaluation (Optimal k) ---")
print("R2 Score:", r2)
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
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
# Note: KNN is a non-parametric, instance-based model.
# It has NO coefficients or tree structure, so standard feature importance does not apply.
# Instead, we use Pearson correlation with the target as a proxy to show which features
# are most related to price.
print("\n--- Feature Importance Analysis (Correlation-based for KNN) ---")
print("Note: KNN has no built-in feature importance. Using Pearson correlation as a proxy.")
correlation = pd.Series(X.values, index=None)
corr_df = pd.DataFrame(X, columns=feature_names)
corr_df['Price'] = y.values
corr_with_price = corr_df.corr()['Price'].drop('Price').abs().sort_values(ascending=False)

most_important = corr_with_price.index[0]
least_important = corr_with_price.index[-1]
print(f"Most Correlated Feature : {most_important} (Correlation: {corr_with_price.iloc[0]:.4f})")
print(f"Reason: This feature has the highest linear correlation with the laptop price.")
print(f"\nLeast Correlated Feature: {least_important} (Correlation: {corr_with_price.iloc[-1]:.4f})")
print(f"Reason: This feature has almost no linear relationship with the laptop price.")

# Feature Correlation Bar Chart (Top 10)
plt.figure(figsize=(12, 6))
corr_with_price.head(10)[::-1].plot(kind='barh', color='steelblue')
plt.xlabel("Absolute Pearson Correlation with Price")
plt.title("Top 10 Features Most Correlated with Price (KNN Proxy Importance)")
plt.tight_layout()
plt.show()

# Visualization of Results
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted (KNN Regression, k={optimal_k})")
plt.show()

plt.figure()
errors = y_test - y_pred
plt.hist(errors, bins=30, edgecolor='black')
plt.title("Error Distribution")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.show()