import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Laptop Price Predictor Pro",
    page_icon="💻",
    layout="wide"
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .price-box {
        background: linear-gradient(135deg, #6e8efb 0%, #a777e3 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(110,142,251,0.3);
        margin: 20px 0;
    }
    .price-value {
        font-size: 48px;
        font-weight: 800;
        color: white;
    }
    .price-label {
        font-size: 16px;
        color: rgba(255,255,255,0.85);
        margin-top: 5px;
    }
    .header-gradient {
        background: linear-gradient(90deg, #00dbde 0%, #fc00ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 42px;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Load data and train models (cached)
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("refined_laptop_data.csv")
    return df

@st.cache_resource
def train_models(df):
    df_encoded = pd.get_dummies(df, drop_first=True)
    X = df_encoded.drop(columns=['Price'])
    y = df_encoded['Price']
    feature_names = X.columns.tolist()

    X_xgb = X.copy()
    X_xgb.columns = X_xgb.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    xgb_feature_names = X_xgb.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)
    X_train_xgb, X_test_xgb, _, _ = train_test_split(X_xgb, y, test_size=0.15, random_state=2)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    trained = {}

    # Linear models on scaled data
    lr = LinearRegression()
    lr.fit(X_train_sc, y_train)
    r2_lr = r2_score(y_test, lr.predict(X_test_sc))
    trained['Linear Regression'] = {'model': lr, 'scaled': True, 'xgb': False, 'r2': r2_lr}

    ridge = RidgeCV(alphas=np.logspace(-4, 4, 20), cv=5)
    ridge.fit(X_train_sc, y_train)
    r2_ridge = r2_score(y_test, ridge.predict(X_test_sc))
    trained['Ridge Regression'] = {'model': ridge, 'scaled': True, 'xgb': False, 'r2': r2_ridge}

    lasso = LassoCV(alphas=np.logspace(-4, 4, 20), cv=5, random_state=42, max_iter=5000)
    lasso.fit(X_train_sc, y_train)
    r2_lasso = r2_score(y_test, lasso.predict(X_test_sc))
    trained['Lasso Regression'] = {'model': lasso, 'scaled': True, 'xgb': False, 'r2': r2_lasso}

    # Tree based models on raw data
    dt = DecisionTreeRegressor(max_depth=10, min_samples_split=5, min_samples_leaf=2, random_state=42)
    dt.fit(X_train, y_train)
    r2_dt = r2_score(y_test, dt.predict(X_test))
    trained['Decision Tree'] = {'model': dt, 'scaled': False, 'xgb': False, 'r2': r2_dt}

    knn = KNeighborsRegressor(n_neighbors=2)
    knn.fit(X_train_sc, y_train)
    r2_knn = r2_score(y_test, knn.predict(X_test_sc))
    trained['KNN (k=2)'] = {'model': knn, 'scaled': True, 'xgb': False, 'r2': r2_knn}

    rf = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    r2_rf = r2_score(y_test, rf.predict(X_test))
    trained['Random Forest'] = {'model': rf, 'scaled': False, 'xgb': False, 'r2': r2_rf}

    xgb = XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                        subsample=0.8, random_state=42, verbosity=0)
    xgb.fit(X_train_xgb, y_train)
    r2_xgb = r2_score(y_test, xgb.predict(X_test_xgb))
    trained['XGBoost'] = {'model': xgb, 'scaled': False, 'xgb': True, 'r2': r2_xgb}

    return trained, feature_names, xgb_feature_names, scaler

df = load_data()

with st.spinner("🔧 Training refined models with advanced features..."):
    trained_models, feature_names, xgb_feature_names, scaler = train_models(df)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown('<p class="header-gradient">💻 Laptop Price Predictor Pro</p>', unsafe_allow_html=True)
st.markdown("### Refined Pre-processing & Advanced Feature Engineering")
st.markdown("---")

# ─────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("##### 🏢 Brand & Type")
    company = st.selectbox("Company", sorted(df['Company'].unique()))
    type_name = st.selectbox("Type", sorted(df['TypeName'].unique()))
    ram = st.selectbox("RAM (GB)", [2, 4, 6, 8, 12, 16, 24, 32, 64], index=3)
    weight = st.number_input("Weight (kg)", 0.5, 5.0, 1.8, 0.1)

with col2:
    st.markdown("##### ⚙️ Internal Specs")
    cpu_brand = st.selectbox("CPU Brand", sorted(df['Cpu brand'].unique()))
    gpu_brand = st.selectbox("GPU Brand", sorted(df['Gpu brand'].unique()))
    os = st.selectbox("Operating System", sorted(df['os'].unique()))
    
    st.markdown("###### Storage (GB)")
    s1, s2 = st.columns(2)
    with s1:
        ssd = st.selectbox("SSD", [0, 8, 128, 256, 512, 1024], index=3)
    with s2:
        hdd = st.selectbox("HDD", [0, 128, 256, 512, 1024, 2048], index=0)

with col3:
    st.markdown("##### 🖥️ Display Specs")
    inches = st.number_input("Screen Size (Inches)", 10.0, 18.0, 15.6, 0.1)
    resolution = st.selectbox("Resolution", [
        '1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', 
        '2880x1800', '2560x1600', '2560x1440', '2304x1440'
    ])
    
    c1, c2 = st.columns(2)
    with c1:
        ips = st.checkbox("IPS Panel")
    with c2:
        touchscreen = st.checkbox("Touchscreen")

st.markdown("---")

# ─────────────────────────────────────────────
# Model Selection
# ─────────────────────────────────────────────
model_col, predict_col = st.columns([1, 1])
with model_col:
    selected_model = st.selectbox("Select Prediction Model", ["🏆 Best Model (Auto)"] + list(trained_models.keys()))

with predict_col:
    st.markdown("##### ")
    predict_button = st.button("🔮 Predict Price", use_container_width=True, type="primary")

if predict_button:
    # 1. Feature Engineering on Input
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / inches
    
    # Storage details
    hybrid = 0
    flash = 0 # simplifying for dashboard
    
    input_df = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_name],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [1 if touchscreen else 0],
        'Ips': [1 if ips else 0],
        'ppi': [ppi],
        'Cpu brand': [cpu_brand],
        'HDD': [hdd],
        'SSD': [ssd],
        'Hybrid': [hybrid],
        'Flash_Storage': [flash],
        'Gpu brand': [gpu_brand],
        'os': [os],
        'Price': [0] # placeholder
    })

    # 2. Encoding
    combined = pd.concat([df, input_df], ignore_index=True)
    combined_encoded = pd.get_dummies(combined, drop_first=True)
    input_encoded = combined_encoded.iloc[[-1]].drop(columns=['Price'])

    # Ensure all feature columns are present and in correct order
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[feature_names]

    # XGBoost version
    input_xgb = input_encoded.copy()
    input_xgb.columns = input_xgb.columns.str.replace(r'[\[\]<>]', '_', regex=True)
    input_xgb = input_xgb[xgb_feature_names]

    # Scaled version
    input_scaled = scaler.transform(input_encoded)

    # 3. Prediction
    st.markdown("---")
    
    def get_price(name, info):
        model = info['model']
        if info['xgb']:
            pred = model.predict(input_xgb)[0]
        elif info['scaled']:
            pred = model.predict(input_scaled)[0]
        else:
            pred = model.predict(input_encoded)[0]
        # Inverse log transformation: exp(x) - 1
        return np.expm1(pred)

    if selected_model == "🏆 Best Model (Auto)":
        predictions = {}
        for name, info in trained_models.items():
            predictions[name] = {'price': get_price(name, info), 'r2': info['r2']}
        
        best_name = max(predictions, key=lambda x: predictions[x]['r2'])
        best_price = predictions[best_name]['price']

        st.markdown(
            f'<div class="price-box">'
            f'<div class="price-label">Best Model: {best_name} (R² = {predictions[best_name]["r2"]:.4f})</div>'
            f'<div class="price-value">₹{best_price:,.0f}</div>'
            f'<div class="price-label">Estimated Laptop Price</div>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("#### 📊 All Model Predictions")
        pred_df = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Predicted Price': [f"₹{v['price']:,.0f}" for v in predictions.values()],
            'R² Score (Log Scale)': [f"{v['r2']:.4f}" for v in predictions.values()],
        })
        st.dataframe(pred_df, use_container_width=True, hide_index=True)

    else:
        info = trained_models[selected_model]
        price = get_price(selected_model, info)
        
        st.markdown(
            f'<div class="price-box">'
            f'<div class="price-label">{selected_model} (R² = {info["r2"]:.4f})</div>'
            f'<div class="price-value">₹{price:,.0f}</div>'
            f'<div class="price-label">Estimated Laptop Price</div>'
            f'</div>',
            unsafe_allow_html=True
        )

    # Input Summary Expander
    with st.expander("📋 Your Configuration Details"):
        c1, c2 = st.columns(2)
        with c1:
            st.write(f"**Company:** {company} | **Type:** {type_name}")
            st.write(f"**CPU:** {cpu_brand} | **GPU:** {gpu_brand}")
            st.write(f"**RAM:** {ram}GB | **Weight:** {weight}kg")
        with c2:
            st.write(f"**OS:** {os}")
            st.write(f"**Storage:** {ssd}GB SSD | {hdd}GB HDD")
            st.write(f"**Display:** {inches}\" {resolution} ({ppi:.1f} PPI)")
            st.write(f"**Features:** {'IPS' if ips else ''} {'Touch' if touchscreen else ''}")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#8892b0; font-size:12px;'>"
    "Laptop Price Predictor Pro | Enhanced Feature Engineering Version | JKLU ML Project"
    "</p>",
    unsafe_allow_html=True
)
