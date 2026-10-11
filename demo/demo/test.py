import streamlit as st
import pandas as pd

#configs
st.title("Table Summarization Generator")
st.set_page_config(layout="wide") # Make the app wide

# basic inputs
st.subheader("Upload CSV File")
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df, use_container_width=True)
    
col1, col2 = st.columns(2)

with col1:
    num_tables = st.number_input("Number of tables", min_value=1, value=1) # k budget

with col2:
    diversity = st.slider("Diversity", 0.0, 1.0, 0.5) # diveresity parameter \theta


def generate_table_summaries(num_tables, diversity, file):
    # TODO: Add logic later
    return f"Tables: {num_tables}\nDiversity: {diversity}\nFile: {file.name if file else 'None'}"

# Popup dialog for results
@st.dialog("Generated Results")
def show_results_dialog(result_text):
    st.write(result_text)

if st.button("Generate Summaries"):
    result = generate_table_summaries(num_tables, diversity, uploaded_file)
    show_results_dialog(result)