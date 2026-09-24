import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Page Setup
st.set_page_config(page_title="SmartStock - Store Manager Portal", layout="wide")

# ==========================================
# SESSION STATE FOR LOGIN
# ==========================================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "shopkeeper_name" not in st.session_state:
    st.session_state.shopkeeper_name = ""
if "selected_store" not in st.session_state:
    st.session_state.selected_store = ""

# ==========================================
# 1. LOGIN SCREEN
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<h1 style='text-align: center;'>🛒 SmartStock: Retail Manager Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>AI-Powered Demand Forecasting & Inventory Allocation for Supermarkets & Groceries</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 🔐 Shopkeeper Login")
        with st.form("login_form"):
            name_input = st.text_input("Shopkeeper / Manager Name", placeholder="E.g., Rajesh Kumar")
            
            # Exactly 3 stores as requested
            store_input = st.selectbox(
                "Select Your Store Branch", 
                [
                    "Store 1: City Center Supermarket (Large)",
                    "Store 2: Express Neighborhood Grocery (Medium)",
                    "Store 3: Suburban Fresh Mart (Small)"
                ]
            )
            
            login_btn = st.form_submit_button("Access Store Dashboard 🚀", use_container_width=True)
            
            if login_btn:
                if name_input.strip() == "":
                    st.warning("Please enter your manager name to proceed.")
                else:
                    st.session_state.logged_in = True
                    st.session_state.shopkeeper_name = name_input
                    st.session_state.selected_store = store_input
                    st.rerun()
    st.stop()

# ==========================================
# 2. GENERATE STORE DATA (Self-Contained)
# ==========================================
@st.cache_data
def load_store_inventory(store_name):
    np.random.seed(42)
    n_rows = 40  # 40 common grocery/supermarket SKUs per store
    
    products = [
        "Basmati Rice (5kg)", "Aashirvaad Atta (10kg)", "Tata Salt (1kg)", 
        "Sunflower Oil (1L)", "Toor Dal (1kg)", "Sugar (1kg)", 
        "Brooke Bond Tea (500g)", "Bru Coffee (200g)", "Maggi Noodles (Pack of 6)", 
        "Amul Butter (500g)", "Fresh Milk (1L)", "Colgate Toothpaste", 
        "Surf Excel Detergent (1kg)", "Lux Soap (Pack of 4)", "Clinic Plus Shampoo",
        "Cadbury Dairy Milk", "Lay's Chips (Large)", "Coca-Cola (2L)", 
        "Whole Wheat Bread", "Eggs (Box of 30)"
    ] * 2
    
    categories = ["Groceries & Staples", "Packaged Foods", "Personal Care", "Household Care", "Beverages"]
    
    data = {
        'Store Branch': store_name,
        'Product Name': products[:n_rows],
        'Category': np.random.choice(categories, n_rows),
        'Current Shelf Stock': np.random.randint(2, 40, n_rows),
        'Item Price (₹)': np.random.uniform(30.0, 650.0, n_rows).round(2),
        'Normal Daily Sales': np.random.randint(5, 25, n_rows),
        'Is Special Promotion Active': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'Is Festival / Holiday Season': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # Simple ML prediction feature simulation for demand
    df['Predicted Daily Demand'] = (
        df['Normal Daily Sales'] 
        + (df['Is Special Promotion Active'] * 8) 
        + (df['Is Festival / Holiday Season'] * 12)
    ).astype(int)
    
    return df

store_data = load_store_inventory(st.session_state.selected_store)

# ==========================================
# 3. SIDEBAR CONTROLS (Practical Shop Controls)
# ==========================================
st.sidebar.markdown(f"👤 **Manager:** {st.session_state.shopkeeper_name}")
st.sidebar.markdown(f"📍 **Branch:** {st.session_state.selected_store}")
if st.sidebar.button("🚪 Logout / Switch Store"):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### 📦 Central Warehouse Allocation")
warehouse_stock_limit = st.sidebar.slider(
    "Total Stock Units Allocated from Central Warehouse Today", 
    min_value=100, 
    max_value=1000, 
    value=400, 
    step=50
)

festival_mode = st.sidebar.checkbox("🎉 Festival / Holiday Rush Mode (Boosts demand across all items)")

if festival_mode:
    store_data['Predicted Daily Demand'] = (store_data['Predicted Daily Demand'] * 1.4).astype(int)

# ==========================================
# 4. CONSTRAINT ALLOCATION ENGINE
# ==========================================
def allocate_stock(df, warehouse_limit):
    df = df.copy()
    # Calculate how many items are needed urgently (Deficit)
    df['Stock Shortage'] = df['Predicted Daily Demand'] - df['Current Shelf Stock']
    df['Stock Shortage'] = df['Stock Shortage'].apply(lambda x: max(0, x))
    
    # Sort by items facing the highest shortage risk
    ranked = df.sort_values(by='Stock Shortage', ascending=False).copy()
    
    shipment = []
    remaining_warehouse = warehouse_limit
    
    for _, row in ranked.iterrows():
        needed = row['Stock Shortage']
        if remaining_warehouse >= needed:
            allocated = needed
            remaining_warehouse -= needed
        else:
            allocated = remaining_warehouse
            remaining_warehouse = 0
        shipment.append(allocated)
        
    ranked['Recommended Restock Quantity'] = shipment
    return ranked

final_allocation = allocate_stock(store_data, warehouse_stock_limit)

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================
st.title(f"📊 Inventory & Restock Dashboard")
st.markdown(f"**Welcome back, {st.session_state.shopkeeper_name}!** Here is your AI-optimized restock plan to prevent empty shelves and overstocking today.")

# Quick Metric Cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Items Monitored", len(final_allocation))
col2.metric("Central Warehouse Pool", f"{warehouse_stock_limit} Units")
col3.metric("Total Daily Predicted Demand", f"{int(final_allocation['Predicted Daily Demand'].sum())} Units")
col4.metric("Items Needing Urgent Restock", len(final_allocation[final_allocation['Stock Shortage'] > 0]))

st.markdown("---")

# Practical Shopkeeper Table
st.markdown("### 🛒 Daily Restock & Delivery Plan for Your Store")
st.markdown("*This list prioritizes items that are about to run out first, ensuring your limited warehouse supply goes where it's needed most.*")

display_table = final_allocation[[
    'Product Name', 'Category', 'Item Price (₹)', 
    'Current Shelf Stock', 'Predicted Daily Demand', 
    'Stock Shortage', 'Recommended Restock Quantity'
]]

st.dataframe(display_table, use_container_width=True, height=400)

# Success confirmation button for store managers
if st.button("✅ Confirm & Send Restock Request to Warehouse", type="primary"):
    st.success(f"Restock request for {warehouse_stock_limit} units successfully transmitted to the central warehouse for {st.session_state.selected_store}!")
