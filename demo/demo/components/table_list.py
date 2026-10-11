import streamlit as st
import pandas as pd

# ITERATION 1 - Original recommendations
df1 = pd.DataFrame(
    {
        "Cost": [1300, 1200, 1000],
    },
    index=["Italian", "French", "Japanese"]
)
df1.index.name = "Cuisines"
df1.columns.name = "Cost"

df2 = pd.DataFrame(
    {
        ("City", "Bengaluru"): [40, 30, 20],
        ("City", "New Delhi"): [20, 10, 40],
        ("City", "Mumbai"): [50, 40, 60],
    },
    index=["Italian", "French", "Japanese"]
)
df2.index.name = "Cuisines"
df2.columns = pd.MultiIndex.from_tuples(df2.columns)

df3 = pd.DataFrame(
    {
        "Rating": [4.4, 4.5, 4.2],
    },
    index=["Bengaluru", "New Delhi", "Mumbai"]
)
df3.index.name = "City"
df3.columns.name = "Rating"

df4 = pd.DataFrame(
    {
        ("Has Online Delivery", "Yes"): [40, 30, 20],
        ("Has Online Delivery", "No"): [20, 10, 40],
    },
    index=["Italian", "French", "Japanese"]
)
df4.index.name = "Cuisines"
df4.columns = pd.MultiIndex.from_tuples(df4.columns)

df5 = pd.DataFrame(
    {
        ("Price Range", "$"): [3.3, 3.6, 4.5],
        ("Price Range", "$$"): [3.5, 4.5, 3.2],
        ("Price Range", "$$$"):[4.5, 3.4, 4.2],
    },
    index=["Italian", "French", "Japanese"]
)
df5.index.name = "Cuisines"
df5.columns = pd.MultiIndex.from_tuples(df5.columns)

# df4 = pd.DataFrame(
#     {
#         ("City", "Bengaluru"): [40, 30, 20],
#         ("City", "New Delhi"): [20, 10, 40],
#         ("City", "Mumbai"): [50, 40, 60],
#     },
#     index=["Italian", "French", "Japanese"]
# )
# df4.index.name = "Cuisines"
# df4.columns = pd.MultiIndex.from_tuples(df2.columns)
iter2_df1 = pd.DataFrame(
    {
        "Cost": [800, 1200, 1400],
    },
    index=["Bengaluru", "New Delhi", "Mumbai"]
)
iter2_df1.index.name = "City"
iter2_df1.columns.name = "Cost"

iter2_df2 = pd.DataFrame(
    {
        ("Price Range", "$"): [3.3, 2.3, 3.2],
        ("Price Range", "$$"): [2.8, 1.0, 3.0],
        ("Price Range", "$$$"):[2.6, 2.5, 2.8],
    },
    index=["Bengaluru", "New Delhi", "Mumbai"]
)
iter2_df2.index.name = "City"
iter2_df2.columns = pd.MultiIndex.from_tuples(iter2_df2.columns)

iter2_df3 = pd.DataFrame(
    {
        ("City", "Bengaluru"): [3.7, 4.0, 3.9],
        ("City", "New Delhi"): [3.8, 4.7, 4.5],
        ("City", "Mumbai"): [4.1, 4.2, 4.3],
    },
    index=["Italian", "French", "Japanese"]
)
iter2_df3.index.name = "Cuisines"
iter2_df3.columns = pd.MultiIndex.from_tuples(iter2_df3.columns)

pivot_tables_iter1 = [
    {
        "query": "AVG(Rating) GROUP BY Price Range, Cuisines",
        "utility": 83,
        "data": df5
    },
    {
        "query": "COUNT(ID) GROUP BY City, Cuisines",
        "utility": 80,
        "data": df2
    },
    {
        "query": "MAX(Cost) GROUP BY Cuisines",
        "utility": 57,
        "data": df1
    },
    {
        "query": "AVG(Rating) GROUP BY City",
        "utility": 75,
        "data": df3
    },
    {
        "query": "COUNT(ID) GROUP BY Cuisines, Has Online Delivery",
        "utility": 70,
        "data": df4
    },
]

df1 = pd.DataFrame(
    {
        "Cost": [1300, 1200, 1000],
    },
    index=["Italian", "French", "Japanese"]
)
df1.index.name = "Cuisines"
df1.columns.name = "Cost"

# ITERATION 2+ - Different recommendations
pivot_tables_iter2 = [
        {
        "query": "AVG(Rating) GROUP BY City, Cuisines",
        "utility": 85,
        "data": iter2_df3
    },
    {
        "query": "AVG(Cost) BY City",
        "utility": 85,
        "data": iter2_df1
    },
    {
        "query": "MIN(Rating) By City, Price Range",
        "utility": 72,
        "data": iter2_df2
    },
]

# Default liked query (not in recommendations)
default_liked_query = {
    "query": "AVG(Rating) GROUP BY Price Range, City",
    "utility": 78,
    "data": pd.DataFrame({
        "Customer_Type": ["Premium", "Regular", "Basic"],
        "Revenue": [45000, 28000, 15000]
    })
}


def render_table_list():
    # Initialize liked/disliked tracking in session state
    # Track by (iteration, query) tuple to persist across iterations
    if "liked_queries" not in st.session_state:
        st.session_state.liked_queries = set()
    if "disliked_queries" not in st.session_state:
        st.session_state.disliked_queries = set()
    
    # Select pivot tables based on iteration
    iteration = st.session_state.get("recommendation_iteration", 0)
    
    if iteration <= 1:
        pivot_tables = pivot_tables_iter1
    else:
        pivot_tables = pivot_tables_iter2
    
    if st.session_state.recommend is False:
        st.subheader(f"Recommendations")
        
    if st.session_state.recommend:
        st.subheader(f"Recommendations ({st.session_state.num_tables})")
        if iteration <=1:
            st.markdown("<p style='margin-top: -10px; margin-bottom: 10px;'>Diversity: 32%</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p style='margin-top: -10px; margin-bottom: 10px;'>Diversity: 30%</p>", unsafe_allow_html=True)
        
        with st.container(border=False, height= 480):
            for idx, pivot_info in enumerate(pivot_tables):
                query = pivot_info['query']
                # Use (iteration, query) tuple as key
                like_key = (iteration, query)
                is_liked = like_key in st.session_state.liked_queries
                is_disliked = like_key in st.session_state.disliked_queries
                
                # Color coded border based on like/dislike status
                border_style = ""
                if is_liked:
                    border_style = "border: 2px solid green;"
                elif is_disliked:
                    border_style = "border: 2px solid red;"
                
                with st.container(border=True):
                    st.markdown(f"<div style='{border_style}'>", unsafe_allow_html=True)
                    
                    st.markdown(f"**Query:** {pivot_info['query']}")
                    st.markdown(f"**Utility Score:** {pivot_info['utility']}%")
                    st.dataframe(pivot_info['data'], use_container_width=True, hide_index=False)
                    
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        if st.button("View Details", key=f"view_{idx}", use_container_width=True):
                            st.session_state.selected_pivot = idx
                    
                    with col2:
                        # Thumbs up button
                        if is_liked:
                            st.markdown('<div style="background-color: #90EE90; padding: 2px 5px; border-radius: 3px; text-align: center; margin-bottom: 2px;">', unsafe_allow_html=True)

                            if st.button("👍", key=f"up_{idx}", use_container_width=True):
                                if like_key in st.session_state.liked_queries:
                                    st.session_state.liked_queries.remove(like_key)
                                else:
                                    st.session_state.liked_queries.add(like_key)
                                    st.session_state.disliked_queries.discard(like_key)
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            if st.button("👍", key=f"up_{idx}", use_container_width=True):
                                st.session_state.liked_queries.add(like_key)
                                st.session_state.disliked_queries.discard(like_key)
                                st.rerun()
                    
                    with col3:
                        # Thumbs down button
                        if is_disliked:
                            st.markdown('<div style="background-color: #FFB6C1; padding: 2px 5px; border-radius: 3px; text-align: center; margin-bottom: 2px;">', unsafe_allow_html=True)
                            if st.button("👎", key=f"down_{idx}", use_container_width=True):
                                if like_key in st.session_state.disliked_queries:
                                    st.session_state.disliked_queries.remove(like_key)
                                else:
                                    st.session_state.disliked_queries.add(like_key)
                                    st.session_state.liked_queries.discard(like_key)
                                st.rerun()
                            st.markdown('</div>', unsafe_allow_html=True)
                        else:
                            if st.button("👎", key=f"down_{idx}", use_container_width=True):
                                st.session_state.disliked_queries.add(like_key)
                                st.session_state.liked_queries.discard(like_key)
                                st.rerun()
                    
                    st.markdown("</div>", unsafe_allow_html=True)


def get_liked_tables():
    """Return list of liked table indices for current iteration"""
    iteration = st.session_state.get("recommendation_iteration", 0)
    if iteration <= 1:
        pivot_tables = pivot_tables_iter1
    else:
        pivot_tables = pivot_tables_iter2
    
    liked_queries = st.session_state.get("liked_queries", set())
    liked_indices = [idx for idx, p in enumerate(pivot_tables) if (iteration, p['query']) in liked_queries]
    return liked_indices


def get_disliked_tables():
    """Return list of disliked table indices for current iteration"""
    iteration = st.session_state.get("recommendation_iteration", 0)
    if iteration <= 1:
        pivot_tables = pivot_tables_iter1
    else:
        pivot_tables = pivot_tables_iter2
    
    disliked_queries = st.session_state.get("disliked_queries", set())
    disliked_indices = [idx for idx, p in enumerate(pivot_tables) if (iteration, p['query']) in disliked_queries]
    return disliked_indices
