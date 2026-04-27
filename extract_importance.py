import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import json

# Load & Prepare Data
df = pd.read_csv("refined_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Price'])
y = df['Price']

# Feature names
feature_names = X.columns.tolist()

# XGBoost-safe column names
X_xgb = X.copy()
X_xgb.columns = X_xgb.columns.str.replace(r'[\[\]<>]', '_', regex=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
X_train_xgb, X_test_xgb, _, _ = train_test_split(X_xgb, y, test_size=0.15, random_state=2)

# Scaled versions for linear models
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

# Models to extract info from
results = {}

# 1. Linear Regression
lr = LinearRegression()
lr.fit(X_train_sc, y_train)
results['Linear Regression'] = dict(zip(feature_names, lr.coef_.tolist()))

# 2. Ridge Regression
ridge = RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5)
ridge.fit(X_train_sc, y_train)
results['Ridge Regression'] = dict(zip(feature_names, ridge.coef_.tolist()))

# 3. Lasso Regression
lasso = LassoCV(alphas=np.logspace(-4, 4, 20), cv=5, random_state=42, max_iter=5000)
lasso.fit(X_train_sc, y_train)
results['Lasso Regression'] = dict(zip(feature_names, lasso.coef_.tolist()))

# 4. Decision Tree
dt = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
dt.fit(X_train, y_train)
results['Decision Tree'] = dict(zip(feature_names, dt.feature_importances_.tolist()))

# 5. Random Forest
rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
results['Random Forest'] = dict(zip(feature_names, rf.feature_importances_.tolist()))

# 6. XGBoost
xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                 random_state=42, verbosity=0)
xgb.fit(X_train_xgb, y_train)
results['XGBoost'] = dict(zip(feature_names, xgb.feature_importances_.tolist()))

# Save to JSON
with open('scratch/feature_analysis.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Feature analysis data saved to scratch/feature_analysis.json")
