import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# Load & Prepare Data
# ─────────────────────────────────────────────
print("Loading data...")
df = pd.read_csv("refined_laptop_data.csv")
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['Price'])
y = df['Price']

# XGBoost-safe column names
X_xgb = X.copy()
X_xgb.columns = X_xgb.columns.str.replace(r'[\[\]<>]', '_', regex=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
X_train_xgb, X_test_xgb, _, _ = train_test_split(X_xgb, y, test_size=0.15, random_state=2)

# Scaled versions for linear models
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ─────────────────────────────────────────────
# Helper: evaluate a model and return metrics
# ─────────────────────────────────────────────
def evaluate(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    
    # Back-transform from log scale
    y_te_real = np.expm1(y_te)
    y_pred_real = np.expm1(y_pred)
    
    r2  = r2_score(y_te, y_pred)
    mae = mean_absolute_error(y_te_real, y_pred_real)
    mse = mean_squared_error(y_te_real, y_pred_real)
    rmse = np.sqrt(mse)
    acc  = 100 - (np.mean(np.abs((y_te_real - y_pred_real) / y_te_real)) * 100)
    return dict(R2=r2, MAE=round(mae, 2), MSE=round(mse, 2), RMSE=round(rmse, 2), Accuracy=round(acc, 2))

# ─────────────────────────────────────────────
# Train all 7 models
# ─────────────────────────────────────────────
results = {}

print("\n[1/6] Training Linear Regression...")
results['Linear Regression'] = evaluate(
    LinearRegression(), X_train_sc, X_test_sc, y_train, y_test)

print("[2/6] Training Ridge Regression...")
results['Ridge Regression'] = evaluate(
    RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5), X_train_sc, X_test_sc, y_train, y_test)

print("[3/6] Training Lasso Regression...")
results['Lasso Regression'] = evaluate(
    LassoCV(alphas=np.logspace(-4, 4, 20), cv=5, random_state=42, max_iter=5000),
    X_train_sc, X_test_sc, y_train, y_test)

print("[4/6] Training Decision Tree...")
results['Decision Tree'] = evaluate(
    DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42),
    X_train, X_test, y_train, y_test)

print("[5/6] Training KNN (k=2, best from Elbow Method)...")
results['KNN (k=2)'] = evaluate(
    KNeighborsRegressor(n_neighbors=2), X_train_sc, X_test_sc, y_train, y_test)

print("[6/6] Training Random Forest...")
results['Random Forest'] = evaluate(
    RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
    X_train, X_test, y_train, y_test)

print("[7/7] Training XGBoost...")
results['XGBoost'] = evaluate(
    XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, subsample=0.8,
                 random_state=42, verbosity=0),
    X_train_xgb, X_test_xgb, y_train, y_test)

# ─────────────────────────────────────────────
# Print Comparison Table
# ─────────────────────────────────────────────
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[['R2', 'MAE', 'RMSE', 'Accuracy']]
comparison_df.index.name = 'Model'

print("\n" + "="*75)
print("               MODEL COMPARISON TABLE")
print("="*75)
print(f"{'Model':<25} {'R2 Score':>10} {'MAE (₹)':>12} {'RMSE (₹)':>12} {'Accuracy %':>12}")
print("-"*75)
for model_name, row in comparison_df.iterrows():
    print(f"{model_name:<25} {row['R2']:>10.4f} {row['MAE']:>12,.2f} {row['RMSE']:>12,.2f} {row['Accuracy']:>11.2f}%")
print("="*75)

best_model = comparison_df['R2'].idxmax()
print(f"\n🏆 Best Model by R2 Score: {best_model} (R2 = {comparison_df.loc[best_model, 'R2']:.4f})")

# ─────────────────────────────────────────────
# Visualization 1: R2 Score Bar Chart
# ─────────────────────────────────────────────
model_names = list(results.keys())
r2_scores    = [results[m]['R2']       for m in model_names]
mae_scores   = [results[m]['MAE']      for m in model_names]
rmse_scores  = [results[m]['RMSE']     for m in model_names]
acc_scores   = [results[m]['Accuracy'] for m in model_names]

colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8BC3', '#8C8C8C']

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Model Comparison: All 7 Models', fontsize=16, fontweight='bold', y=1.01)

# Plot 1 - R2 Score
ax1 = axes[0, 0]
bars = ax1.bar(model_names, r2_scores, color=colors, edgecolor='black', linewidth=0.7)
ax1.set_title('R2 Score (Higher is Better)', fontweight='bold')
ax1.set_ylabel('R2 Score')
ax1.set_ylim(0, 1.1)
ax1.axhline(y=0.8, color='green', linestyle='--', linewidth=1.2, label='Good threshold (0.8)')
ax1.legend(fontsize=8)
ax1.tick_params(axis='x', rotation=20)
for bar, val in zip(bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Plot 2 - MAE
ax2 = axes[0, 1]
bars2 = ax2.bar(model_names, mae_scores, color=colors, edgecolor='black', linewidth=0.7)
ax2.set_title('MAE - Mean Absolute Error (Lower is Better)', fontweight='bold')
ax2.set_ylabel('MAE (₹)')
ax2.tick_params(axis='x', rotation=20)
for bar, val in zip(bars2, mae_scores):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'₹{val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 3 - RMSE
ax3 = axes[1, 0]
bars3 = ax3.bar(model_names, rmse_scores, color=colors, edgecolor='black', linewidth=0.7)
ax3.set_title('RMSE - Root Mean Squared Error (Lower is Better)', fontweight='bold')
ax3.set_ylabel('RMSE (₹)')
ax3.tick_params(axis='x', rotation=20)
for bar, val in zip(bars3, rmse_scores):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             f'₹{val:,.0f}', ha='center', va='bottom', fontsize=8, fontweight='bold')

# Plot 4 - Accuracy
ax4 = axes[1, 1]
bars4 = ax4.bar(model_names, acc_scores, color=colors, edgecolor='black', linewidth=0.7)
ax4.set_title('Accuracy % (Higher is Better)', fontweight='bold')
ax4.set_ylabel('Accuracy (%)')
ax4.tick_params(axis='x', rotation=20)
for bar, val in zip(bars4, acc_scores):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
             f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# Visualization 2: R2 Score Progression Chart
# ─────────────────────────────────────────────
plt.figure(figsize=(12, 6))
plt.plot(model_names, r2_scores, marker='o', markersize=10, linewidth=2.5,
         color='royalblue', markerfacecolor='red', markeredgewidth=2)
for i, (name, val) in enumerate(zip(model_names, r2_scores)):
    plt.annotate(f'{val:.3f}', (i, val), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=10, fontweight='bold', color='darkblue')
plt.axhline(y=0.8, color='green', linestyle='--', linewidth=1.5, label='Good Performance Threshold (R2=0.8)')
plt.title('Model Sophistication vs Performance (R2 Score Progression)', fontsize=14, fontweight='bold')
plt.xlabel('Models (Simple → Sophisticated)', fontsize=12)
plt.ylabel('R2 Score', fontsize=12)
plt.ylim(0.5, 1.05)
plt.xticks(rotation=15)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print("\nComparison complete!")
