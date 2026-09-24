import pandas as pd
import numpy as np
import streamlit as st

# Page Setup & Clean Light Theme Styling
st.set_page_config(page_title="SmartStock - Supermarket Manager Portal", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .main { background-color: #f8fafc; color: #1e293b; padding: 2rem; }
        .stSidebar { background-color: #e2e8f0; padding: 1.5rem 1rem; }
        
        div[data-testid="stMetric"] {
            background-color: #ffffff;
            border: 1px solid #cbd5e1;
            padding: 20px;
            border-radius: 12px;
            color: #1e293b;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            margin-bottom: 1rem;
        }
        div[data-testid="stMetric"] label { color: #64748b !important; font-size: 0.9rem !important; }
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #0f172a !important; font-size: 1.8rem !important; }

        .stButton button { 
            border-radius: 8px; 
            font-weight: 600; 
            background-color: #2563eb; 
            color: white; 
            border: none;
            padding: 0.6rem 1.2rem;
        }
        .stButton button:hover { background-color: #1d4ed8; color: white; }
        
        h1, h2, h3 { color: #0f172a !important; font-family: sans-serif; }
        .element-container { margin-bottom: 1rem; }
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
# 1. LOGIN SCREEN
# ==========================================
if not st.session_state.logged_in:
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center; color: #2563eb; margin-bottom: 0.2rem;'>SmartStock AI</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #64748b; margin-bottom: 2rem;'>Inventory & Demand Optimization Portal</p>", unsafe_allow_html=True)
        
        with st.container():
            st.markdown("### Manager Authentication")
            with st.form("login_form"):
                name_input = st.text_input("Store Manager Name", placeholder="Enter your full name")
                
                store_input = st.selectbox(
                    "Select Store Branch", 
                    [
                        "Store 1: Fresh Fruit & Produce Market",
                        "Store 2: City Center Supermarket (Groceries)",
                        "Store 3: Express Neighborhood Mart (Daily Needs)"
                    ]
                )
                
                st.markdown("<br>", unsafe_allow_html=True)
                login_btn = st.form_submit_button("Launch Portal", use_container_width=True)
                
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
# 2. GENERATE STORE DATA (Branch Specific)
# ==========================================
@st.cache_data
def load_store_inventory(store_name):
    np.random.seed(42)
    n_rows = 35 
    
    # Customize products based on the selected store branch
    if "Store 1" in store_name:
        products = [
            "Fresh Apple (1kg)", "Valencia Orange (1kg)", "Cavendish Banana (Dozen)", 
            "Alphonso Mango (1kg)", "Seedless Grapes (500g)", "Fresh Papaya (1pc)", 
            "Ripe Pineapple (1pc)", "Watermelon (1pc)", "Pomegranate (1kg)", 
            "Sweet Lemon Mosambi (1kg)", "Guava (1kg)", "Kiwi Fruit (Pack of 3)"
        ] * 3
        categories = ["Fresh Fruits", "Organic Produce", "Seasonal Fruits"]
    elif "Store 2" in store_name:
        products = [
            "Basmati Rice (5kg)", "Aashirvaad Atta (10kg)", "Tata Salt (1kg)", 
            "Sunflower Oil (1L)", "Toor Dal (1kg)", "Sugar (1kg)", 
            "Brooke Bond Tea (500g)", "Bru Coffee (200g)", "Maggi Noodles (Pack of 6)"
        ] * 4
        categories = ["Groceries & Staples", "Packaged Foods"]
    else:
        products = [
            "Amul Butter (500g)", "Fresh Milk (1L)", "Colgate Toothpaste", 
            "Surf Excel Detergent (1kg)", "Lux Soap (Pack of 4)", "Whole Wheat Bread", "Eggs (Box of 30)"
        ] * 5
        categories = ["Dairy & Bakery", "Personal Care", "Household Care"]
    
    data = {
        'Store Branch': store_name,
        'Product Name': products[:n_rows],
        'Category': np.random.choice(categories, n_rows),
        'Current Shelf Stock': np.random.randint(2, 45, n_rows),
        'Item Price (Rs.)': np.random.uniform(30.0, 350.0, n_rows).round(2),
        'Normal Daily Sales': np.random.randint(6, 25, n_rows),
        'Special Promotion Active': np.random.choice([0, 1], n_rows, p=[0.7, 0.3]),
        'Festival Season Active': np.random.choice([0, 1], n_rows, p=[0.8, 0.2]),
    }
    
    df = pd.DataFrame(data)
    
    df['Predicted Daily Demand'] = (
        df['Normal Daily Sales'] 
        + (df['Special Promotion Active'] * 10) 
        + (df['Festival Season Active'] * 14)
    ).astype(int)
    
    return df

store_data = load_store_inventory(st.session_state.selected_store)

# ==========================================
# 3. SIDEBAR CONTROLS
# ==========================================
st.sidebar.markdown(f"### Manager: {st.session_state.shopkeeper_name}")
st.sidebar.markdown(f"**Branch:** {st.session_state.selected_store}")

st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("Logout / Switch Branch", use_container_width=True):
    st.session_state.logged_in = False
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Warehouse Configuration")

warehouse_stock_limit = st.sidebar.slider(
    "Central Warehouse Supply Limit (Units)", 
    min_value=150, 
    max_value=1200, 
    value=500, 
    step=50
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
festival_mode = st.sidebar.toggle("Festival / Holiday Surge Mode", value=False)
flash_sale = st.sidebar.toggle("Flash Sale / Weekend Rush Active", value=False)

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
    
    def calculate_urgency(row):
        if row['Current Shelf Stock'] <= 3:
            return 5
        elif row['Stock Shortage'] > 10:
            return 4
        elif row['Stock Shortage'] > 5:
            return 3
        elif row['Stock Shortage'] > 0:
            return 2
        else:
            return 1
            
    ranked['Urgency Index'] = ranked.apply(calculate_urgency, axis=1)
    return ranked

final_allocation = allocate_stock(store_data, warehouse_stock_limit)

# ==========================================
# 5. MAIN DASHBOARD UI
# ==========================================
st.title("Store Inventory & Restock Dashboard")
st.markdown("Real-time predictive stock allocation ensuring limited warehouse shipments prevent stockouts and optimize inventory distribution.")

st.markdown("<br>", unsafe_allow_html=True)

# Metrics Row with padding
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total SKUs Tracked", len(final_allocation))
col2.metric("Warehouse Pool Limit", f"{warehouse_stock_limit} Units")
col3.metric("Total Predicted Demand", f"{int(final_allocation['Predicted Daily Demand'].sum())} Units")
high_urgency_count = len(final_allocation[final_allocation['Urgency Index'] >= 4])
col4.metric("High Urgency Items", high_urgency_count)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### Prioritized Restock Queue")

# Filter View Options
filter_option = st.radio(
    "Filter View by Urgency:", 
    ["All Items", "Urgency Level 5 (Critical)", "Urgency Level 4 (High)", "Urgency Level 1-3 (Normal)"], 
    horizontal=True
)

if filter_option == "Urgency Level 5 (Critical)":
    display_df = final_allocation[final_allocation['Urgency Index'] == 5]
elif filter_option == "Urgency Level 4 (High)":
    display_df = final_allocation[final_allocation['Urgency Index'] == 4]
elif filter_option == "Urgency Level 1-3 (Normal)":
    display_df = final_allocation[final_allocation['Urgency Index'] <= 3]
else:
    display_df = final_allocation

st.markdown("<br>", unsafe_allow_html=True)

# Formatted Table Display
st.dataframe(
    display_df[[
        'Product Name', 'Category', 'Item Price (Rs.)', 
        'Current Shelf Stock', 'Predicted Daily Demand', 
        'Stock Shortage', 'Recommended Restock Quantity', 'Urgency Index'
    ]],
    use_container_width=True,
    height=420
)

st.markdown("<br>", unsafe_allow_html=True)

# Action Section
col_a, col_b = st.columns([2, 1])
with col_a:
    st.info("System Note: Items featuring a higher Urgency Index are prioritized automatically to ensure high-demand inventory is replenished first under active central warehouse caps.")
with col_b:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Confirm and Dispatch Order", use_container_width=True):
        st.success(f"Successfully ordered {warehouse_stock_limit} units for {st.session_state.selected_store}.")
