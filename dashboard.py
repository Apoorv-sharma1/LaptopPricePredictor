import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor - Model Analysis",
    page_icon="💻",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS for premium look
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: #00d4ff;
    }
    .metric-label {
        font-size: 14px;
        color: #8892b0;
        margin-top: 5px;
    }
    .header-gradient {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 800;
    }
    .stSelectbox > div > div { background-color: #1a1a2e; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load and prepare data (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    # Use refined data (with log-transformed prices)
    df = pd.read_csv("refined_laptop_data.csv")
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=['Price'])
    y = df_encoded['Price']
    feature_names = X.columns.tolist()

    # Clean column names for XGBoost
    X_xgb = X.copy()
    X_xgb.columns = X_xgb.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    xgb_feature_names = X_xgb.columns.tolist()

    # Align splitting with scripts: test_size=0.15, random_state=2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
    X_train_xgb, X_test_xgb, _, _ = train_test_split(X_xgb, y, test_size=0.15, random_state=2)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return (df, X_train, X_test, y_train, y_test,
            X_train_sc, X_test_sc,
            X_train_xgb, X_test_xgb,
            feature_names, xgb_feature_names, scaler)

(df, X_train, X_test, y_train, y_test,
 X_train_sc, X_test_sc,
 X_train_xgb, X_test_xgb,
 feature_names, xgb_feature_names, scaler) = load_data()

# ─────────────────────────────────────────────
# Train models (cached)
# ─────────────────────────────────────────────
@st.cache_resource
def train_all_models():
    models = {}

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    models['Linear Regression'] = {'model': lr, 'scaled': True, 'xgb': False,
                                    'color': '#4C72B0', 'icon': '📈'}

    # Ridge Regression
    ridge = RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5)
    ridge.fit(X_train_sc, y_train)
    models['Ridge Regression'] = {'model': ridge, 'scaled': True, 'xgb': False,
                                   'color': '#55A868', 'icon': '📊'}

    # Lasso Regression
    lasso = LassoCV(alphas=np.logspace(-4, 4, 20), cv=5, random_state=42, max_iter=5000)
    lasso.fit(X_train_sc, y_train)
    models['Lasso Regression'] = {'model': lasso, 'scaled': True, 'xgb': False,
                                   'color': '#C44E52', 'icon': '📉'}

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    dt.fit(X_train, y_train)
    models['Decision Tree'] = {'model': dt, 'scaled': False, 'xgb': False,
                                'color': '#8172B2', 'icon': '🌳'}

    # KNN
    knn = KNeighborsRegressor(n_neighbors=2)
    knn.fit(X_train_sc, y_train)
    models['KNN (k=2)'] = {'model': knn, 'scaled': True, 'xgb': False,
                            'color': '#937860', 'icon': '🔍'}

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = {'model': rf, 'scaled': False, 'xgb': False,
                                'color': '#DA8BC3', 'icon': '🌲'}

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                        subsample=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train_xgb, y_train)
    models['XGBoost'] = {'model': xgb, 'scaled': False, 'xgb': True,
                          'color': '#8C8C8C', 'icon': '🚀'}

    return models

with st.spinner("Training all 7 models... Please wait."):
    models = train_all_models()

# ─────────────────────────────────────────────
# KNN Elbow Method data (cached)
# ─────────────────────────────────────────────
@st.cache_data
def compute_knn_elbow():
    k_range = range(1, 21)
    rmse_values = []
    for k in k_range:
        knn = KNeighborsRegressor(n_neighbors=k)
        knn.fit(X_train_sc, y_train)
        y_pred_k = knn.predict(X_test_sc)
        rmse_k = np.sqrt(mean_squared_error(y_test, y_pred_k))
        rmse_values.append(rmse_k)
    optimal_k = list(k_range)[np.argmin(rmse_values)]
    return list(k_range), rmse_values, optimal_k

knn_k_range, knn_rmse_values, knn_optimal_k = compute_knn_elbow()

# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def get_predictions(model_info):
    model = model_info['model']
    if model_info['xgb']:
        return model.predict(X_test_xgb)
    elif model_info['scaled']:
        return model.predict(X_test_sc)
    else:
        return model.predict(X_test)

def get_metrics(y_true, y_pred):
    # R2 is calculated on the scale the model was trained on (log scale)
    r2 = r2_score(y_true, y_pred)
    
    # Convert back to real prices for MAE, MSE, RMSE, and Accuracy
    y_true_real = np.expm1(y_true)
    y_pred_real = np.expm1(y_pred)
    
    mae = mean_absolute_error(y_true_real, y_pred_real)
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    
    # Accuracy formula: 100 - MAPE
    mape = np.mean(np.abs((y_true_real - y_pred_real) / y_true_real)) * 100
    acc = 100 - mape
    
    return {'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'Accuracy': acc}

def get_feature_importance(model_name, model_info):
    model = model_info['model']
    if hasattr(model, 'feature_importances_'):
        names = xgb_feature_names if model_info['xgb'] else feature_names
        imp = model.feature_importances_
        return pd.DataFrame({'Feature': names, 'Importance': imp}).sort_values('Importance', ascending=False)
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_)
        return pd.DataFrame({'Feature': feature_names, 'Importance': imp}).sort_values('Importance', ascending=False)
    else:
        return None

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="header-gradient">💻 Laptop Price Predictor</p>', unsafe_allow_html=True)
st.markdown("### Model Analysis Dashboard")
st.markdown("---")

# ─────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔬 Individual Model Analysis", "📊 Compare All Models"])

# ═══════════════════════════════════════════════
# TAB 1: Individual Model
# ═══════════════════════════════════════════════
with tab1:
    col_select, col_info = st.columns([1, 2])
    with col_select:
        model_names = list(models.keys())
        selected = st.selectbox("Choose a Model", model_names, index=5)

    model_info = models[selected]
    y_pred = get_predictions(model_info)
    metrics = get_metrics(y_test, y_pred)

    with col_info:
        st.markdown(f"### {model_info['icon']} {selected}")

    # Metric Cards
    st.markdown("#### 📋 Performance Metrics")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("R² Score", f"{metrics['R2']:.4f}")
    with c2:
        st.metric("MAE", f"₹{metrics['MAE']:,.0f}")
    with c3:
        st.metric("MSE", f"{metrics['MSE']:,.0f}")
    with c4:
        st.metric("RMSE", f"₹{metrics['RMSE']:,.0f}")
    with c5:
        st.metric("Accuracy", f"{metrics['Accuracy']:.1f}%")

    st.markdown("---")

    # Charts Row
    chart1, chart2 = st.columns(2)

    with chart1:
        st.markdown("#### 🎯 Actual vs Predicted Price")
        # Convert to real prices for chart
        y_test_real = np.expm1(y_test)
        y_pred_real = np.expm1(y_pred)
        
        fig1, ax1 = plt.subplots(figsize=(8, 6))
        ax1.scatter(y_test_real, y_pred_real, alpha=0.4, color=model_info['color'], edgecolors='white', linewidth=0.5, s=50)
        ax1.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2, label='Perfect Prediction')
        ax1.set_xlabel("Actual Price (₹)", fontsize=12)
        ax1.set_ylabel("Predicted Price (₹)", fontsize=12)
        ax1.legend()
        ax1.set_facecolor('#0e1117')
        fig1.patch.set_facecolor('#0e1117')
        ax1.tick_params(colors='white')
        ax1.xaxis.label.set_color('white')
        ax1.yaxis.label.set_color('white')
        ax1.spines['bottom'].set_color('#333')
        ax1.spines['left'].set_color('#333')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        st.pyplot(fig1)

    with chart2:
        st.markdown("#### 📊 Error Distribution")
        # Convert to real prices for error calculation
        errors_real = np.expm1(y_test) - np.expm1(y_pred)
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.hist(errors_real, bins=30, color=model_info['color'], edgecolor='white', alpha=0.8)
        ax2.axvline(x=0, color='red', linestyle='--', lw=2, label='Zero Error')
        ax2.set_xlabel("Prediction Error (₹)", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.legend()
        ax2.set_facecolor('#0e1117')
        fig2.patch.set_facecolor('#0e1117')
        ax2.tick_params(colors='white')
        ax2.xaxis.label.set_color('white')
        ax2.yaxis.label.set_color('white')
        ax2.spines['bottom'].set_color('#333')
        ax2.spines['left'].set_color('#333')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        st.pyplot(fig2)

    # ─── Hyperparameter Tuning Section ───
    st.markdown("---")
    st.markdown("#### ⚙️ Hyperparameter Tuning")

    if selected == 'KNN (k=2)':
        st.markdown("##### Elbow Method: How k=2 was chosen")
        st.caption("We tested k values from 1 to 20 and measured the RMSE for each. "
                   "The optimal k is the one with the lowest error.")
        fig_elbow, ax_elbow = plt.subplots(figsize=(10, 5))
        ax_elbow.plot(knn_k_range, knn_rmse_values, marker='o', linestyle='--',
                      color='#00d4ff', markerfacecolor='#ff4444', markeredgecolor='white', markeredgewidth=1.5, markersize=8)
        ax_elbow.axvline(x=knn_optimal_k, color='lime', linestyle='--', lw=2,
                         label=f'Optimal k = {knn_optimal_k}')
        ax_elbow.set_xlabel('Value of k', fontsize=12, color='white')
        ax_elbow.set_ylabel('RMSE', fontsize=12, color='white')
        ax_elbow.set_title('Elbow Method: k vs RMSE', fontsize=14, color='white', fontweight='bold')
        ax_elbow.set_xticks(knn_k_range)
        ax_elbow.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        ax_elbow.set_facecolor('#0e1117')
        fig_elbow.patch.set_facecolor('#0e1117')
        ax_elbow.tick_params(colors='white')
        ax_elbow.spines['bottom'].set_color('#333')
        ax_elbow.spines['left'].set_color('#333')
        ax_elbow.spines['top'].set_visible(False)
        ax_elbow.spines['right'].set_visible(False)
        ax_elbow.grid(True, alpha=0.15)
        plt.tight_layout()
        st.pyplot(fig_elbow)

        # Show the RMSE table for each k
        elbow_df = pd.DataFrame({'k': knn_k_range, 'RMSE': [f'{v:,.2f}' for v in knn_rmse_values]})
        with st.expander("📋 View RMSE for all k values"):
            st.dataframe(elbow_df, hide_index=True)

    elif selected == 'Ridge Regression':
        ridge_model = model_info['model']
        st.info(f"**Optimal Alpha (λ) found by RidgeCV:** `{ridge_model.alpha_:.4f}`")
        st.caption("RidgeCV tested 20 alpha values (from 0.0001 to 10000) using 5-fold cross-validation "
                   "and automatically selected the alpha that gave the best performance.")

    elif selected == 'Lasso Regression':
        lasso_model = model_info['model']
        st.info(f"**Optimal Alpha (λ) found by LassoCV:** `{lasso_model.alpha_:.4f}`")
        st.caption("LassoCV tested 20 alpha values using 5-fold cross-validation. "
                   "Lasso also performs feature selection by shrinking unimportant coefficients to exactly zero.")
        non_zero = np.sum(lasso_model.coef_ != 0)
        total = len(lasso_model.coef_)
        st.warning(f"Lasso kept **{non_zero}** out of **{total}** features (eliminated {total - non_zero} features).")

    elif selected == 'Decision Tree':
        dt_model = model_info['model']
        st.info(f"**Parameters:** max_depth={dt_model.max_depth}, "
                f"min_samples_split={dt_model.min_samples_split}, "
                f"min_samples_leaf={dt_model.min_samples_leaf}")
        st.caption("These parameters control tree complexity to prevent overfitting. "
                   "max_depth limits how deep the tree can grow.")

    elif selected == 'Random Forest':
        rf_model = model_info['model']
        st.info(f"**Parameters:** n_estimators={rf_model.n_estimators}, max_depth={rf_model.max_depth}")
        st.caption(f"Random Forest builds {rf_model.n_estimators} different decision trees and averages their predictions. "
                   f"This reduces variance and prevents overfitting compared to a single tree.")

    elif selected == 'XGBoost':
        xgb_model = model_info['model']
        st.info(f"**Parameters:** n_estimators={xgb_model.n_estimators}, max_depth={xgb_model.max_depth}, "
                f"learning_rate={xgb_model.learning_rate}, subsample={xgb_model.subsample}")
        st.caption("XGBoost uses Gradient Boosting where each tree corrects the mistakes of the previous one. "
                   "The learning_rate controls how much each tree contributes.")

    elif selected == 'Linear Regression':
        st.info("**No hyperparameters to tune.** Linear Regression fits a direct linear equation "
                "to the data using Ordinary Least Squares (OLS).")
        st.caption("This is the simplest baseline model. It assumes a linear relationship between features and price.")

    # Feature Importance
    st.markdown("---")
    st.markdown("#### 🏆 Feature Importance Analysis")
    importance_df = get_feature_importance(selected, model_info)

    if importance_df is not None:
        top_10 = importance_df.head(10)
        bottom_3 = importance_df.tail(3)

        col_chart, col_text = st.columns([2, 1])
        with col_chart:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            bars = ax3.barh(top_10['Feature'][::-1], top_10['Importance'][::-1],
                           color=model_info['color'], edgecolor='white', linewidth=0.5)
            ax3.set_xlabel("Importance Score", fontsize=12, color='white')
            ax3.set_title("Top 10 Most Important Features", fontsize=14, color='white', fontweight='bold')
            ax3.set_facecolor('#0e1117')
            fig3.patch.set_facecolor('#0e1117')
            ax3.tick_params(colors='white')
            ax3.spines['bottom'].set_color('#333')
            ax3.spines['left'].set_color('#333')
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig3)

        with col_text:
            most = importance_df.iloc[0]
            least = importance_df.iloc[-1]
            st.success(f"**Most Important:** {most['Feature']}")
            st.caption(f"Score: {most['Importance']:.4f}")
            st.info("This feature has the strongest influence on predicting laptop prices.")

            st.error(f"**Least Important:** {least['Feature']}")
            st.caption(f"Score: {least['Importance']:.6f}")
            st.info("This feature has almost no impact on the price prediction.")
    else:
        st.warning("KNN is an instance-based model and does not have traditional feature importance. "
                   "It makes predictions based on the distance to the nearest neighbors.")

# ═══════════════════════════════════════════════
# TAB 2: Compare All Models
# ═══════════════════════════════════════════════
with tab2:
    st.markdown("### 📊 All Models Comparison")

    # Build comparison data
    all_metrics = {}
    for name, info in models.items():
        pred = get_predictions(info)
        all_metrics[name] = get_metrics(y_test, pred)

    comp_df = pd.DataFrame(all_metrics).T
    comp_df.index.name = 'Model'

    # Highlight best model
    best_model = comp_df['R2'].idxmax()
    st.success(f"🏆 **Best Model: {best_model}** with R² = {comp_df.loc[best_model, 'R2']:.4f}")

    # Table
    display_df = comp_df.copy()
    display_df['R2'] = display_df['R2'].apply(lambda x: f"{x:.4f}")
    display_df['MAE'] = display_df['MAE'].apply(lambda x: f"₹{x:,.0f}")
    display_df['MSE'] = display_df['MSE'].apply(lambda x: f"{x:,.0f}")
    display_df['RMSE'] = display_df['RMSE'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Accuracy'] = display_df['Accuracy'].apply(lambda x: f"{x:.2f}%")
    st.dataframe(display_df, width='stretch')

    st.markdown("---")

    # Comparison Charts
    chart_col1, chart_col2 = st.columns(2)

    model_list = list(all_metrics.keys())
    colors = [models[m]['color'] for m in model_list]

    with chart_col1:
        st.markdown("#### R² Score Comparison")
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        r2_vals = [all_metrics[m]['R2'] for m in model_list]
        bars = ax4.bar(model_list, r2_vals, color=colors, edgecolor='white', linewidth=0.7)
        ax4.axhline(y=0.8, color='lime', linestyle='--', linewidth=1.5, label='Good Threshold (0.8)')
        ax4.set_ylim(0, 1.1)
        for bar, val in zip(bars, r2_vals):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{val:.3f}', ha='center', fontsize=9, fontweight='bold', color='white')
        ax4.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
        ax4.set_facecolor('#0e1117')
        fig4.patch.set_facecolor('#0e1117')
        ax4.tick_params(colors='white', axis='x', rotation=20)
        ax4.tick_params(colors='white', axis='y')
        ax4.spines['bottom'].set_color('#333')
        ax4.spines['left'].set_color('#333')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig4)

    with chart_col2:
        st.markdown("#### Accuracy Comparison")
        fig5, ax5 = plt.subplots(figsize=(10, 6))
        acc_vals = [all_metrics[m]['Accuracy'] for m in model_list]
        bars = ax5.bar(model_list, acc_vals, color=colors, edgecolor='white', linewidth=0.7)
        for bar, val in zip(bars, acc_vals):
            ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold', color='white')
        ax5.set_facecolor('#0e1117')
        fig5.patch.set_facecolor('#0e1117')
        ax5.tick_params(colors='white', axis='x', rotation=20)
        ax5.tick_params(colors='white', axis='y')
        ax5.spines['bottom'].set_color('#333')
        ax5.spines['left'].set_color('#333')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig5)

    # Progression Chart
    st.markdown("---")
    st.markdown("#### 📈 Model Sophistication vs Performance")
    fig6, ax6 = plt.subplots(figsize=(14, 5))
    r2_vals = [all_metrics[m]['R2'] for m in model_list]
    ax6.plot(model_list, r2_vals, marker='o', markersize=12, linewidth=2.5,
             color='#00d4ff', markerfacecolor='#ff4444', markeredgewidth=2, markeredgecolor='white')
    for i, val in enumerate(r2_vals):
        ax6.annotate(f'{val:.3f}', (i, val), textcoords="offset points",
                     xytext=(0, 15), ha='center', fontsize=11, fontweight='bold', color='#00d4ff')
    ax6.axhline(y=0.8, color='lime', linestyle='--', linewidth=1.5, label='Good Threshold (R²=0.8)')
    ax6.set_xlabel("Models (Simple → Sophisticated)", fontsize=12, color='white')
    ax6.set_ylabel("R² Score", fontsize=12, color='white')
    ax6.set_ylim(0.5, 1.05)
    ax6.legend(facecolor='#1a1a2e', edgecolor='#333', labelcolor='white')
    ax6.grid(True, alpha=0.15)
    ax6.set_facecolor('#0e1117')
    fig6.patch.set_facecolor('#0e1117')
    ax6.tick_params(colors='white', axis='x', rotation=15)
    ax6.tick_params(colors='white', axis='y')
    ax6.spines['bottom'].set_color('#333')
    ax6.spines['left'].set_color('#333')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig6)

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8892b0; font-size:12px;'>"
    "Laptop Price Predictor | ML Project by Apoorv, Lakshya & Rohit Soni | JKLU Semester 4"
    "</p>",
    unsafe_allow_html=True
)
