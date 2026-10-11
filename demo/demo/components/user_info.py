import streamlit as st
import pandas as pd
import textwrap
from components.table_list import *

def user_preferences():
    # Add CSS for hover effects
    st.markdown("""
    <style>
    .query-content {
        cursor: pointer;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 12px;
    }
    .query-content:hover {
        filter: brightness(0.95);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.subheader("User Preferences")
    st.markdown(f"**User:** {st.session_state.user_id}")
    if st.session_state.login_time:
        st.markdown(f"**Login time:** {st.session_state.login_time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    # if st.button("Logout", key="logout_btn"):
        # st.session_state.logged_in = False
        # st.session_state.user_id = ""
        # st.rerun()
    
    # Check if data is uploaded
    has_data = st.session_state.get("uploaded_file") is not None
    
    # Select pivot tables based on iteration
    iteration = st.session_state.get("recommendation_iteration", 0)
    if iteration <= 1:
        pivot_tables = pivot_tables_iter1
    else:
        pivot_tables = pivot_tables_iter2
    
    liked = get_liked_tables()
    disliked = get_disliked_tables()
    
    with st.container(border=True, height=300):
        st.markdown("#### 👍 Liked Tables")
        # Show default liked query only when data is uploaded
        if has_data:
            st.markdown(f"""
            <div class="query-content" style="background-color: #d4edda;">
                {default_liked_query['query']}<br>
            </div>
            """, unsafe_allow_html=True)
        
        # Get all liked queries (not just current iteration)
        liked_queries = st.session_state.get("liked_queries", set())
        if liked_queries:
            for like_key in sorted(liked_queries):
                iteration_num, query_text = like_key
                st.markdown(f"""
                <div class="query-content" style="background-color: #d4edda;">
                    {query_text}<br>
                </div>
                """, unsafe_allow_html=True)
    
    with st.container(border=True, height=300):
        st.markdown("#### 👎 Disliked Tables")
        # Get all disliked queries (not just current iteration)
        disliked_queries = st.session_state.get("disliked_queries", set())
        if disliked_queries:
            for dislike_key in sorted(disliked_queries):
                iteration_num, query_text = dislike_key
                st.markdown(f"""
                <div class="query-content" style="background-color: #f8d7da;">
                    {query_text}<br>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("<p style='color: #888;'><i>No disliked tables yet</i></p>", unsafe_allow_html=True)
