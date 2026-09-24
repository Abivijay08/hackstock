import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Inventory Demand & Allocation Engine", layout="wide")

st.title("📦 Inventory-Constrained Demand Forecasting & Allocation Engine")
st.markdown("### HTH-ML-06: Retail Supply Chain Optimization MVP")

# ==========================================
# 1. AUTO-GENERATE DATASET (No CSV Required!)
# ==========================================
@st.cache_data
def generate_synthetic_retail_data():
    np.random.seed(42)
    n_rows = 500
    
    stores = ['Store_A', 'Store_B', 'Store_C', 'Store_D', 'Store_E']
    products = ['SKU_101', 'SKU_102', 'SKU_103', 'SKU_104', 'SKU_105', 'SKU_106']
    categories = ['Electronics', 'Apparel', 'Groceries', 'Home & Living']
    
    data = {
        'Store ID': np.random.choice(stores, n_rows),
        'Product ID': np.random.choice(products, n_rows),
        'Category': np.random.choice(categories, n_rows),
        'Inventory Level': np.random.randint(5, 150, n_rows),
        'Price': np.random.uniform(10.0, 1500.0, n_rows),
        'Promotion': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'Holiday': np.random.choice([0, 1], n_rows, p=[0.85, 0.15]),
        'Holiday_Promotion': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Simulate realistic historical sales target influenced by features
    df['Units Sold'] = (
        10 
        + (df['Promotion'] * 25) 
        + (df['Holiday'] * 30) 
        + (df['Holiday_Promotion'] * 50) 
        + (df['Inventory Level'] * 0.2) 
        + np.random.normal(0, 5, n_rows)
    ).clip(lower=1).astype(int)
    
    return df

with st.spinner("Initializing dataset and training demand model..."):
    df = generate_synthetic_retail_data()

    # ==========================================
    # 2. TRAIN DEMAND FORECASTING MODEL
    # ==========================================
    features = ['Inventory Level', 'Price', 'Promotion', 'Holiday', 'Holiday_Promotion']
    target = 'Units Sold'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    df['Forecasted_Demand'] = model.predict(df[features])

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Allocation Controls")
total_inventory_pool = st.sidebar.slider(
    "Total Warehouse Inventory Pool (Units)", 
    min_value=500, 
    max_value=10000, 
    value=3000, 
    step=250
)

simulate_spike = st.sidebar.checkbox("🚀 Simulate Mid-Demo Promotional Holiday Spike (1.5x Demand)")

if simulate_spike:
    df['Forecasted_Demand'] = df['Forecasted_Demand'] * 1.5

# ==========================================
# 4. ALLOCATION OPTIMIZER
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

allocation_results = run_allocation_optimizer(df, total_inventory_pool)

# ==========================================
# 5. UI METRICS & TABLE
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs Managed", len(allocation_results))
col2.metric("Allocated Inventory Pool", f"{total_inventory_pool} Units")
col3.metric("Total Forecasted Demand", f"{int(allocation_results['Forecasted_Demand'].sum())} Units")

st.markdown("---")
st.markdown("### 📊 Prioritized Store-Level Allocation Queue")

display_cols = ['Store ID', 'Product ID', 'Category', 'Inventory Level', 'Holiday', 'Holiday_Promotion', 'Forecasted_Demand', 'Deficit', 'Allocated_Stock']
st.dataframe(allocation_results[display_cols].head(25), use_container_width=True)
