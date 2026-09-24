import os
import pandas as pd
import streamlit as st
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Page Configuration
st.set_page_config(page_title="Inventory Demand & Allocation Engine", layout="wide")

st.title("📦 Inventory-Constrained Demand Forecasting & Allocation Engine")
st.markdown("### HTH-ML-06: Retail Supply Chain Optimization MVP")

# ==========================================
# 1. LOAD & CACHE DATA VIA KAGGLEHUB
# ==========================================
@st.cache_data
def load_and_prepare_data():
    path = kagglehub.dataset_download("anirudhchauhan/retail-store-inventory-forecasting-dataset")
    csv_file_path = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_file_path = os.path.join(root, file)
                break
    
    df = pd.read_csv(csv_file_path)
    df.columns = df.columns.str.strip()
    
    # Add dummy holiday columns if not present in raw data
    if 'Holiday' not in df.columns:
        df['Holiday'] = 0 
    if 'Holiday_Promotion' not in df.columns:
        df['Holiday_Promotion'] = 0
        
    return df

with st.spinner("Downloading dataset and training demand model..."):
    df = load_and_prepare_data()

    # ==========================================
    # 2. TRAIN DEMAND FORECASTING MODEL
    # ==========================================
    features = ['Inventory Level', 'Price', 'Promotion', 'Holiday', 'Holiday_Promotion']
    target = 'Units Sold'

    model_df = df.dropna(subset=features + [target]).copy()
    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    model_df['Forecasted_Demand'] = model.predict(model_df[features])

# ==========================================
# 3. SIDEBAR CONTROLS (INTERACTIVE DEMO)
# ==========================================
st.sidebar.header("⚙️ Allocation Controls")
total_inventory_pool = st.sidebar.slider(
    "Total Warehouse Inventory Pool (Units)", 
    min_value=500, 
    max_value=20000, 
    value=5000, 
    step=500
)

# Bonus feature simulation toggle
simulate_spike = st.sidebar.checkbox("🚀 Simulate Mid-Demo Promotional Holiday Spike (1.5x Demand)")

if simulate_spike:
    model_df['Forecasted_Demand'] = model_df['Forecasted_Demand'] * 1.5

# ==========================================
# 4. INVENTORY CONSTRAINT ALLOCATION OPTIMIZER
# ==========================================
def run_allocation_optimizer(df_subset, total_pool):
    df_subset = df_subset.copy()
    df_subset['Deficit'] = df_subset['Forecasted_Demand'] - df_subset['Inventory Level']
    ranked_df = df_subset.sort_values(by='Deficit', ascending=False).copy()
    
    allocated_units = []
    remaining_pool = total_pool
    
    for _, row in ranked_df.iterrows():
        needed = max(0, row['Forecasted_Demand'] - row['Inventory Level'])
        if remaining_pool >= needed:
            allocated = needed
            remaining_pool -= needed
        else:
            allocated = remaining_pool
            remaining_pool = 0
            
        allocated_units.append(allocated)
        
    ranked_df['Allocated_Stock'] = allocated_units
    return ranked_df

allocation_results = run_allocation_optimizer(model_df, total_inventory_pool)

# ==========================================
# 5. DASHBOARD LAYOUT & METRICS
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs Managed", len(allocation_results))
col2.metric("Allocated Inventory Pool", f"{total_inventory_pool} Units")
col3.metric("Total Forecasted Demand", f"{int(allocation_results['Forecasted_Demand'].sum())} Units")

st.markdown("---")
st.markdown("### 📊 Prioritized Store-Level Allocation Queue")
st.dataframe(
    allocation_results[[
        'Store ID', 'Product ID', 'Category', 'Inventory Level', 
        'Holiday', 'Holiday_Promotion', 'Forecasted_Demand', 'Deficit', 'Allocated_Stock'
    ]].head(25),
    use_container_width=True
)
