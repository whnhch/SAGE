import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def radar_from_scores(scores: dict):
    labels = list(scores.keys())
    values = list(scores.values())

    fig = go.Figure(
        data=go.Scatterpolar(
            r=values + [values[0]],
            theta=labels + [labels[0]],
            fill="toself"
        )
    )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1])
        ),
        showlegend=False,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    return fig


# def visualize_pivot_table(df):
#     st.markdown("### Pivot Table")

#     st.caption("Query: GROUP BY City, AGG = AVG(Price)")

#     st.dataframe(df, use_container_width=True)

#     st.markdown("**Utility:** 0.8")

#     tab1, tab2 = st.tabs(["Insightfulness", "Interpretability"])


def visualize_pivot_table(df):

    st.subheader("Pivot Table Visualization")

    selected = st.session_state.get("selected_pivot", None)

    if selected is None:
        st.empty()
        return

    # Import pivot_tables based on iteration
    from components.table_list import pivot_tables_iter1, pivot_tables_iter2
    
    iteration = st.session_state.get("recommendation_iteration", 0)
    pivot_tables = pivot_tables_iter1 if iteration <= 1 else pivot_tables_iter2
    
    with st.container(border=False, height=500):
        pivot_info = pivot_tables[selected]
        
        st.markdown(f"#### {pivot_info['query']}")
        st.dataframe(pivot_info['data'], use_container_width=True)
        st.markdown(f"#### Utility: {pivot_info['utility']}%")

        # Show Insightfulness and Interpretability side by side
        col_insight, col_divider, col_interpret = st.columns([1, 0.08, 1])

        with col_insight:
            st.markdown("**Insightfulness**")
            
            iteration = st.session_state.get("recommendation_iteration", 0)

            if iteration<=1:
                insightfulness_scores = {
                    "Attribute Significance": 0.87,
                    "Informativeness": 0.75,
                    "Trend": 0.55,
                    "Surprise": 0.60,
                    }
            else:
                insightfulness_scores = {
                    "Attribute Significance": 1.00,
                    "Informativeness": 0.70,
                    "Trend": 0.00,
                    "Surprise": 0.00,
                    }
                                
            overall_score = insightfulness_scores["Attribute Significance"] * max(insightfulness_scores["Informativeness"], insightfulness_scores["Trend"],insightfulness_scores["Surprise"])
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown("**Insightfulness**")
            with c2:
                st.progress(overall_score)
            with c3:
                st.markdown(f"{overall_score*100:.0f}%")
            
            st.markdown("")  # spacer
            
            # Individual metric scores
            for name, val in insightfulness_scores.items():
                c1, c2, c3 = st.columns([2, 1, 1])
                with c1:
                    st.markdown(name)
                with c2:
                    st.progress(val)
                with c3:
                    st.markdown(f"{val*100:.0f}%")
            
            st.markdown("")  # spacer before radar
            fig = radar_from_scores(insightfulness_scores)
            st.plotly_chart(fig, use_container_width=True)

        with col_divider:
            st.markdown("""
                <div style="border-left: 3px solid #999; height: 100%; margin: 0 auto;"></div>
            """, unsafe_allow_html=True)

        with col_interpret:
            st.markdown("**Interpretability**")
            
            if iteration<=1:
                interpretability_scores = {
                    "Density": 1.00,
                    "Clarity1": 0.00,
                    "Clarity2": 0.00,
                    "Semantic Validity": 0.90,
                    "Conciseness": 0.95,
                    "Clarity3": 0.00,
                }
                interpretability_scores_radar = {
                    "Density": 1.00,
                    "Semantic Validity": 0.90,
                    "Conciseness": 0.95,
                }
            else:
                interpretability_scores = {
                    "Density": 1.00,
                    "Clarity1": 0.00,
                    "Clarity2": 0.00,
                    "Semantic Validity": 1.00,
                    "Conciseness": 1.00,
                    "Clarity3": 0.00,
                }
                interpretability_scores_radar = {
                    "Density": 1.00,
                    "Semantic Validity": 1.00,
                    "Conciseness": 1.00,
                }
            
            # Overall score (average of visible metrics)
            overall_score = sum(interpretability_scores_radar.values()) / len(interpretability_scores_radar)
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown("**Interpretability**")
            with c2:
                st.progress(overall_score)
            with c3:
                st.markdown(f"{overall_score*100:.0f}%")
            
            st.markdown("")  # spacer
            
            # Individual metric scores
            for i, (name, val) in enumerate(interpretability_scores.items()):
                if i == 5 or i==1 or i==2:  
                    st.markdown("")  
                else:
                    c1, c2, c3 = st.columns([2, 1, 1])
                    with c1:
                        st.markdown(f"{name}")
                    with c2:
                        st.progress(val)
                    with c3:
                        st.markdown(f"{val*100:.0f}%")
            
            st.markdown("")  # spacer before radar
            fig = radar_from_scores(interpretability_scores_radar)
            st.plotly_chart(fig, use_container_width=True)

            