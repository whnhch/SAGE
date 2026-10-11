import streamlit as st
import pandas as pd
import textwrap


def render_data_preview(df=None, font_family=None):
    # Outer gray container
    gray_bg_style = """
    <div style="
        background-color:#f0f2f6; 
        padding:30px; 
        border-radius:8px; 
        display:flex; 
        justify-content:center;
        ">
    """

    # default font if none provided
    if font_family is None:
        font_family = "Inter, Arial, sans-serif"

    if df is None:
        # Placeholder white card
        inner_content = textwrap.dedent(f"""
        <div style="
            background-color:white; 
            padding:30px; 
            border-radius:8px; 
            width:95%; 
            max-width:1400px; 
            text-align:center; 
            color:#888;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            font-size:16px;
            font-family: {font_family};
            ">
            No dataset uploaded yet.
        </div>
        """)
    else:
        # Calculate column widths based on header lengths
        col_widths = {}
        for col in df.columns:
            # Calculate minimum width needed for the column header (roughly 8px per character)
            header_width = max(len(str(col)) * 8, 80)  # Minimum 80px
            col_widths[col] = header_width
        
        # Generate column width styles
        col_style = ""
        for i, (col, width) in enumerate(col_widths.items()):
            col_style += f"table th:nth-child({i+1}), table td:nth-child({i+1}) {{ min-width: {width}px; white-space: nowrap; }}\n"
        
        # Table HTML with styling and scrollable if large
        table_html = df.head(3).to_html(index=False, border=0)
        inner_content = textwrap.dedent(f"""
        <div style="
            background-color:white; 
            padding:20px; 
            border-radius:8px; 
            width:95%; 
            max-width:1920px; 
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
            overflow-x:auto;
            font-family: {font_family};
            ">
            <style>
                table {{ width: 100%; border-collapse: collapse; color: black; font-size:14px; font-family: {font_family}; table-layout: auto; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; color: black; font-weight: 600; }}
                {col_style}
            </style>
            {table_html}
        </div>
        """)

    # Combine gray container + white card
    full_html = gray_bg_style + inner_content + "</div>"

    # Use st.components.v1.html to render raw HTML safely (avoids Markdown code-block escaping)
    try:
        st.components.v1.html(full_html, width=1920, height=300, scrolling=True)
    except Exception:
        st.markdown(full_html, unsafe_allow_html=True)