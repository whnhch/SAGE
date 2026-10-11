import streamlit as st
import pandas as pd

def set_params(df):   
    st.subheader("Parameters")
    st.markdown("Diversity (%)")
    
    # -------------------------
    # Diversity stepper
    # -------------------------
    col1, col2, col3 = st.columns([1.5, 2, 1.5])
    with col1:
        if st.button("➖", key="div_minus", use_container_width=True):
            st.session_state.diversity = max(10, st.session_state.diversity - 10)
            st.rerun()

    with col2:
        st.number_input(
            "Diversity (%)",
            min_value=10,
            max_value=100,
            value=int(st.session_state.diversity),
            step=10,
            key="diversity_input",
            label_visibility="collapsed"
        )

    with col3:
        if st.button("➕", key="div_plus", use_container_width=True):
            st.session_state.diversity = min(100, st.session_state.diversity + 10)
            st.rerun()

    # -------------------------
    # Number of tables stepper
    # -------------------------
    st.markdown("Number of tables")
    
    col4, col5, col6 = st.columns([1.5, 2, 1.5])

    with col4:
        if st.button("➖", key="num_minus"):
            st.session_state.num_tables = max(1, st.session_state.num_tables - 1)
            st.rerun()

    with col5:
        st.session_state.num_tables = st.number_input(
            "Number of tables",
            min_value=1,
            max_value=10,
            value=st.session_state.num_tables,
            step=1,
            label_visibility="collapsed"
        )

    with col6:
        if st.button("➕", key="num_plus"):
            st.session_state.num_tables = min(10, st.session_state.num_tables + 1)
            st.rerun()
    
    # Attributes selection
    if df is None:
        attributes_options = []
    else:
        attributes_options = df.columns.tolist()  # all column names as options
    selected_attributes = st.multiselect("Preffered Attributes", options=attributes_options)

    # if selected_attributes:
    #     st.markdown("**Selected attributes:** " + ", ".join(selected_attributes))
    # else:
    #     st.markdown("*No attributes selected*")
        
    return st.session_state.diversity, st.session_state.num_tables, selected_attributes