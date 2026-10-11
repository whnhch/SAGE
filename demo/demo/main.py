import streamlit as st
import pandas as pd
from datetime import datetime
import random
from components.data_preview import *
from components.params import *
from components.table_list import *
from components.pivot_table import visualize_pivot_table
from components.user_info import *

st.cache_data.clear()
st.cache_resource.clear()

st.set_page_config(layout="wide")

# Initialize session state for login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_id" not in st.session_state:
    st.session_state.user_id = ""
if "login_time" not in st.session_state:
    st.session_state.login_time = None

# Login dialog
@st.dialog("Login to SAGE")
def login_dialog():
    st.write("Please enter your user ID to continue")
    user_id = st.text_input("User ID", key="login_user_id")
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("Login", use_container_width=True):
            if user_id:
                st.session_state.user_id = user_id
                st.session_state.logged_in = True
                # Generate random time in December 2025
                random_day = random.randint(1, 31)
                random_hour = random.randint(0, 23)
                random_minute = random.randint(0, 59)
                random_second = random.randint(0, 59)
                st.session_state.login_time = datetime(2025, 12, random_day, random_hour, random_minute, random_second)
                st.rerun()
            else:
                st.error("Please enter a user ID")

# Show login dialog if not logged in
if not st.session_state.logged_in:
    login_dialog()
    st.stop()

# Make uploader narrow and add scrollable sections CSS
st.markdown("""
<style>
body, html {
    font-size: 14px;
}

div[data-testid="stFileUploader"] {
    width: 400px;
}

/* Scrollable sections with fixed height */
.scrollable-section {
    height: 600px;
    overflow-y: auto;
    padding-right: 10px;
}

/* Custom scrollbar styling */
.scrollable-section::-webkit-scrollbar {
    width: 8px;
}
.scrollable-section::-webkit-scrollbar-track {
    background: #f1f1f1;
}
.scrollable-section::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}
.scrollable-section::-webkit-scrollbar-thumb:hover {
    background: #555;
}

/* Make all columns have equal fixed height */
[data-testid="column"] {
    padding: 0 8px;
    height: 600px !important;
    overflow: hidden !important;
}

[data-testid="column"]:first-child {
    padding-left: 0;
}

[data-testid="column"]:last-child {
    padding-right: 0;
}

/* Constrain column content */
[data-testid="column"] > div {
    height: 100% !important;
    display: flex !important;
    flex-direction: column !important;
}

/* Set fixed height for each column's container */
[data-testid="column"]:nth-child(1) > div > div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100% !important;
    overflow: hidden !important;
}

[data-testid="column"]:nth-child(2) > div > div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100% !important;
    overflow: hidden !important;
}

[data-testid="column"]:nth-child(3) > div > div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100% !important;
    overflow: hidden !important;
}

[data-testid="column"]:nth-child(4) > div > div[data-testid="stVerticalBlockBorderWrapper"] {
    height: 100% !important;
    overflow: hidden !important;
}

/* Gray background for all border wrapper containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #f5f5f5 !important;
    border-radius: 8px !important;
    border: 1px solid #e0e0e0 !important;
    padding: 20px !important;
    box-sizing: border-box !important;
}

/* Make inner content transparent */
div[data-testid="stVerticalBlockBorderWrapper"] > div {
    background: none !important;
}

</style>
""", unsafe_allow_html=True)

# Inline row
with st.container():
    col_title, col_upload = st.columns([0.1, 0.6])  # tiny first column to hug the title

    with col_title:
        st.title("SAGE")

    with col_upload:
        uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")
        st.session_state.uploaded_file = uploaded_file

# -------------------------
# Dataset preview section
# -------------------------
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    # Standardize city names
    if 'City' in df.columns:
        df['City'] = df['City'].replace('Bangalore', 'Bengaluru')
        df = df[df['City'].isin(['Mumbai', 'New Delhi', 'Bengaluru'])]
else:
    df = None

render_data_preview(df)

# Create main columns
col1, col2, col3, col4 = st.columns([1.3, 2.5, 5, 2])

if "selected_pivot" not in st.session_state:
    st.session_state.selected_pivot = None
# -------------------------
# Parameters section
# -------------------------
with col1:
    with st.container(border=True, height=505):
        if "diversity" not in st.session_state:
            st.session_state.diversity = 20

        if "num_tables" not in st.session_state:
            st.session_state.num_tables = 1
        
        if "recommend" not in st.session_state:
            st.session_state.recommend = False
        
        if "recommendation_iteration" not in st.session_state:
            st.session_state.recommendation_iteration = 0
            
        diversity, num_tables, selected_attributes = set_params(df)
        
    # Recommendation button
    with st.container(border=True):
        left_spacer, center_col, right_spacer = st.columns([0.1, 2, 0.1])
        with center_col:
            if st.button("Recommend", key="recommend_btn", use_container_width=True):
                st.session_state.recommendation_iteration += 1
                st.session_state.recommend = True
                st.session_state.selected_pivot = None

# -------------------------
# Recommendation list section
# -------------------------
with col2:
    with st.container(border=True):
        render_table_list()

# -------------------------
# Visualize section
# -------------------------
with col3:
    with st.container(border=True):
        visualize_pivot_table(df)

with col4:
    with st.container(border=True, height=580):
        user_preferences()