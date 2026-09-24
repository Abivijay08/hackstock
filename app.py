import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor

# Page Setup & Dark Theme Styling
st.set_page_config(page_title="SmartStock - Supermarket Manager Portal", layout="wide", initial_sidebar_state="expanded")

# Custom Dark Theme CSS
st.markdown("""
    <style>
        /* Main Background & Font Color */
        .main { background-color: #0e1117; color: #fafafa; }
        .stSidebar { background-color: #161b22; }
        
        /* Metric Cards Styling */
        div[data-testid="stMetric"] {
            background-color: #1f2937;
            border: 1px solid #374151;
            padding: 15px;
            border-radius: 10px;
            color: #ffffff;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }
        div[data-testid="stMetric"] label { color: #9ca3af !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f3f4f6 !important; }

        /* Buttons & Inputs */
        .stButton button { 
            border-radius: 8px; 
            font-weight: bold; 
            background-color: #2563eb; 
            color: white; 
            border: none;
        }
        .stButton button:hover { background-color: #1d4ed8; }
        
        /* Headers */
        h1, h2, h3 { color: #f9fafb !important; }
    </style>
""", unsafe_allow_html=True)

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
# 1. INTERACTIVE LOGIN SCREEN (Dark Theme)
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #60a5fa;'>🛒 SmartStock AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #9ca3af;'>Next-Gen Inventory & Demand Optimization for Retailers</p>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("### 🔐 Manager Portal Login")
            with st.form("login_form"):
                name_input = st.text_input("Store Manager Name", placeholder="E.g., Abishek V")
                
                # Exactly 3 store branches
                store_input = st.selectbox(
                    "Select Store Branch", 
                    [
                        "Store 1: City Center Supermarket (Large)",
                        "Store 2: Express Neighborhood Grocery (Medium)",
                        "Store 3: Suburban Fresh Mart (Small)"
                    ]
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                login_btn = st.form_submit_button("Launch Dashboard 🚀", use_container_width=True)
                
                if login_btn:
                    if name_input.strip() == "":
                        st.warning("Please enter your name to proceed.")
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
    n_rows = 45 
    
    products = [
        "Basmati Rice (5kg)", "Aashirvaad Atta (10kg)", "Tata Salt (1kg)", 
        "Sunflower Oil (1L)", "Toor Dal (1kg)", "Sugar (1kg)", 
        "Brooke Bond Tea (500g)", "Bru Coffee (200g)", "Maggi Noodles (Pack of 6)", 
        "Amul Butter (500g)", "Fresh Milk (1L)", "Colgate Toothpaste", 
        "Surf Excel Detergent (1kg)", "Lux Soap (Pack of 4)", "Clinic Plus Shampoo",
        "Cadbury Dairy Milk", "Lay's Chips (Large)", "Coca-Cola (2L)", 
        "Whole Wheat Bread", "Eggs (Box of 30)"
    ] * 3
    
    categories = ["Groceries & Staples", "Packaged Foods", "Personal Care", "Household Care", "Beverages"]
    
    data = {
        'Store Branch': store_name,
        'Product Name': products[:n_rows],
        'Category': np.random.choice(categories, n_rows),
        'Current Shelf Stock': np.random.randint(2, 45, n_rows),
        'Item Price (₹)': np.random.uniform(30.0, 750.0, n_rows).round(2),
        'Normal Daily Sales': np.random.randint(6, 25, n_rows),
        'Is Special Promotion Active': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'Is Festival / Holiday Season': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    # ML Demand Simulation
    df['Predicted Daily Demand'] = (
        df['Normal Daily Sales'] 
        + (df['Is Special Promotion Active'] * 10) 
        + (df['Is Festival / Holiday Season'] * 14)
    ).astype(int)
    
    return df

store_data = load_store_inventory(st.session_state.selected_store)

# ==========================================
# 3. INTERACTIVE SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown(f"### 👋 Welcome, {st.session_state.shopkeeper_name}")
st.sidebar.markdown(f"📍 **Branch:** `{st.session_state.selected_store}`")

if st.sidebar.button("🚪 Logout / Switch Branch", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Warehouse & Allocation Settings")

warehouse_stock_limit = st.sidebar.slider(
    "📦 Central Warehouse Supply Limit (Units)", 
    min_value=150, 
    max_value=1200, 
    value=500, 
    step=50
)

festival_mode = st.sidebar.toggle("🎉 Festival / Holiday Surge Mode", value=False)
flash_sale = st.sidebar.toggle("⚡ Flash Sale / Weekend Rush Active", value=False)

if festival_mode:
    store_data['Predicted Daily Demand'] = (store_data['Predicted Daily Demand'] * 1.3).astype(int)
if flash_sale:
    store_data['Predicted Daily Demand'] = (store_data['Predicted Daily Demand'] * 1.25).astype(int)

# ==========================================
# 4. CONSTRAINT ALLOCATION OPTIMIZER ENGINE
# ==========================================
def allocate_stock(df, warehouse_limit):
    df = df.copy()
    df['Stock Shortage'] = df['Predicted Daily Demand'] - df['Current Shelf Stock']
    df['Stock Shortage'] = df['Stock Shortage'].apply(lambda x: max(0, x))
    
    # Priority sorting based on scarcity gap
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
    
    def get_status(row):
        if row['Current Shelf Stock'] <= 3:
            return "🚨 Critical Stockout Risk"
        elif row['Stock Shortage'] > 5:
            return "⚠️ Restock Recommended"
        else:
            return "✅ Stock Adequate"
            
    ranked['Inventory Status'] = ranked.apply(get_status, axis=1)
    return ranked

final_allocation = allocate_stock(store_data, warehouse_stock_limit)

# ==========================================
# 5. MAIN INTERACTIVE DASHBOARD
# ==========================================
st.title("📊 SmartStore Manager Dashboard")
st.markdown("Real-time predictive stock allocation ensuring your limited warehouse shipments prevent empty shelves and maximize sales.")

# Metrics Row
col1, col2, col3, col4 = st.columns(4)
col1.metric("📦 Total SKUs Tracked", len(final_allocation))
col2.metric("🚛 Warehouse Pool", f"{warehouse_stock_limit} Units")
col3.metric("🔥 Total Demand", f"{int(final_allocation['Predicted Daily Demand'].sum())} Units")
critical_count = len(final_allocation[final_allocation['Inventory Status'] == "🚨 Critical Stockout Risk"])
col4.metric("🚨 Critical Alerts", critical_count)

st.markdown("---")

# Filter view option
st.markdown("### 🛒 Prioritized Restock Queue")
filter_option = st.radio(
    "Filter View:", 
    ["All Items", "🚨 Critical Stockout Risk Only", "⚠️ Restock Recommended Only"], 
    horizontal=True
)

if filter_option == "🚨 Critical Stockout Risk Only":
    display_df = final_allocation[final_allocation['Inventory Status'] == "🚨 Critical Stockout Risk"]
elif filter_option == "⚠️ Restock Recommended Only":
    display_df = final_allocation[final_allocation['Inventory Status'] == "⚠️ Restock Recommended"]
else:
    display_df = final_allocation

# Formatted Table Display
st.dataframe(
    display_df[[
        'Product Name', 'Category', 'Item Price (₹)', 
        'Current Shelf Stock', 'Predicted Daily Demand', 
        'Stock Shortage', 'Recommended Restock Quantity', 'Inventory Status'
    ]],
    use_container_width=True,
    height=420
)

# Interactive Restock Submission
col_a, col_b = st.columns([2, 1])
with col_a:
    st.info("💡 **AI Recommendation Note:** Items facing high shortages are prioritized automatically under your central warehouse supply limits.")
with col_b:
    if st.button("✅ Confirm & Dispatch Order", use_container_width=True):
        st.balloons()
        st.success(f"Successfully ordered {warehouse_stock_limit} units for {st.session_state.selected_store}!")
