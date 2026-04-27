import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np

# Load the data
with open('scratch/feature_analysis.json', 'r') as f:
    data = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(data)

# Normalize data
df_norm = df.copy()
for col in df_norm.columns:
    if 'Regression' in col or 'Linear' in col:
        df_norm[col] = df_norm[col].abs()
    max_val = df_norm[col].max()
    if max_val > 0:
        df_norm[col] = df_norm[col] / max_val

# Top 15 features
df_norm['Avg'] = df_norm.mean(axis=1)
top_features = df_norm.sort_values(by='Avg', ascending=False).head(15).index.tolist()
df_plot = df_norm.loc[top_features].drop(columns=['Avg'])

# Plotting with Matplotlib
fig, ax = plt.subplots(figsize=(14, 10))

models = df_plot.columns.tolist()
features = df_plot.index.tolist()
n_models = len(models)
n_features = len(features)

bar_width = 0.8 / n_models
y = np.arange(n_features)

colors = plt.cm.get_cmap('tab10', n_models)

for i, model in enumerate(models):
    ax.barh(y + i*bar_width, df_plot[model], bar_width, label=model, color=colors(i))

ax.set_yticks(y + bar_width * (n_models - 1) / 2)
ax.set_yticklabels(features)
ax.invert_yaxis()  # labels read top-to-bottom

ax.set_xlabel('Importance Score (Normalized 0-1)', fontsize=12)
ax.set_title('Combined Feature Importance Comparison (Normalized)', fontsize=16, fontweight='bold')
ax.legend(title='Models', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('scratch/combined_feature_importance.png', dpi=300, bbox_inches='tight')
print("Plot saved to scratch/combined_feature_importance.png")

# Heatmap part removed because seaborn is not available.
