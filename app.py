import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Inventory Demand & Allocation Engine", layout="wide")

st.title("📦 Inventory-Constrained Demand Forecasting & Allocation Engine")
st.markdown("### HTH-ML-06: Retail Supply Chain Optimization MVP")

# ==========================================
# 1. LOAD & CLEAN DATA
# ==========================================
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv("retail_store_inventory.csv")
    # Clean up column names (strip trailing spaces and standardize)
    df.columns = df.columns.str.strip()
    return df

try:
    df = load_and_prepare_data()
except Exception as e:
    st.error(f"Error loading CSV file: {e}")
    st.stop()

# Helper to find matching column names case-insensitively
def find_col(possible_names):
    for col in df.columns:
        if col.lower() in [p.lower() for p in possible_names]:
            return col
    return None

# Map expected features to actual CSV columns dynamically
col_inventory = find_col(['Inventory Level', 'inventory_level', 'stock'])
col_price = find_col(['Price', 'price', 'unit_price'])
col_promo = find_col(['Promotion', 'promotion', 'promo'])
col_target = find_col(['Units Sold', 'units_sold', 'sales'])
col_store = find_col(['Store ID', 'store_id', 'store'])
col_product = find_col(['Product ID', 'product_id', 'sku'])
col_category = find_col(['Category', 'category'])

# Add holiday columns if missing
if 'Holiday' not in df.columns:
    df['Holiday'] = 0 
if 'Holiday_Promotion' not in df.columns:
    df['Holiday_Promotion'] = 0

features = [col_inventory, col_price, col_promo, 'Holiday', 'Holiday_Promotion']
target = col_target

# Validate columns exist
missing_cols = [c for c in features + [target] if c is None]
if missing_cols:
    st.error(f"Could not find matching columns in your CSV for: {missing_cols}. Available columns are: {list(df.columns)}")
    st.stop()

# ==========================================
# 2. TRAIN DEMAND FORECASTING MODEL
# ==========================================
with st.spinner("Training demand forecasting model..."):
    model_df = df.dropna(subset=features + [target]).copy()
    X = model_df[features]
    y = model_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    model_df['Forecasted_Demand'] = model.predict(model_df[features])

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.header("⚙️ Allocation Controls")
total_inventory_pool = st.sidebar.slider(
    "Total Warehouse Inventory Pool (Units)", 
    min_value=500, 
    max_value=20000, 
    value=5000, 
    step=500
)

simulate_spike = st.sidebar.checkbox("🚀 Simulate Mid-Demo Promotional Holiday Spike (1.5x Demand)")

if simulate_spike:
    model_df['Forecasted_Demand'] = model_df['Forecasted_Demand'] * 1.5

# ==========================================
# 4. ALLOCATION OPTIMIZER
# ==========================================
def run_allocation_optimizer(df_subset, total_pool):
    df_subset = df_subset.copy()
    df_subset['Deficit'] = df_subset['Forecasted_Demand'] - df_subset[col_inventory]
    ranked_df = df_subset.sort_values(by='Deficit', ascending=False).copy()
    
    allocated_units = []
    remaining_pool = total_pool
    
    for _, row in ranked_df.iterrows():
        needed = max(0, row['Forecasted_Demand'] - row[col_inventory])
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
# 5. UI METRICS & TABLE
# ==========================================
col1, col2, col3 = st.columns(3)
col1.metric("Total SKUs Managed", len(allocation_results))
col2.metric("Allocated Inventory Pool", f"{total_inventory_pool} Units")
col3.metric("Total Forecasted Demand", f"{int(allocation_results['Forecasted_Demand'].sum())} Units")

st.markdown("---")
st.markdown("### 📊 Prioritized Store-Level Allocation Queue")

display_cols = [c for c in [col_store, col_product, col_category, col_inventory, 'Holiday', 'Holiday_Promotion', 'Forecasted_Demand', 'Deficit', 'Allocated_Stock'] if c is not None]
st.dataframe(allocation_results[display_cols].head(25), use_container_width=True)
