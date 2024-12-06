import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
import seaborn as sns
from scipy.stats import trim_mean
from math import pi
from dotenv import load_dotenv
import os
from udfs import *
# load_dotenv()

# Title of the app
st.title('IBNR Loss Reserves Usecase')

# Sidebar navigation
st.sidebar.title("Tool Capabilities")
option = st.sidebar.radio('Choose a section:', ['Input Dataset', 'Average LDFs', 
                                                'LDFs Summary Report', 'Actuary Methods for IBNR', 
                                                'Ultimate Claims & IBNR Reserves Summary Report',
                                                'Diagnostic Metrics Evaluation'])

# Ensure session state for dataset exists
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if option == 'Input Dataset':
    st.write("Welcome to the IBNR Loss Reserves application. Please upload your dataset to begin!")
    uploaded_file = st.file_uploader("Upload your CSV dataset file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.uploaded_df = df  # Store the dataset in session state
        st.write("Data Preview:")
        st.dataframe(df.head())

        try:
            with st.spinner('Processing data...'):
                st.session_state.uploaded_df = calculate_derived_columns(df)
                st.success("Calculations for derived columns completed.")

            if st.button('Show Age-to-Age Factors Heatmap'):
                with st.spinner('Generating heatmap...'):
                    calculate_and_plot_ataf(st.session_state.uploaded_df)
        except Exception as e:
            st.error(f"Failed to process file: {e}")

else:
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        calculate_ataf(st.session_state.uploaded_df)

        if option == 'Average LDFs':
            st.write("We have 7 averaging methods available to calculate LDFs from the ATAF. Please select one from the dropdown below:")
            averaging_methods = ['Select an averaging method','Simple Moving Average', 'Volume Weighted Average', 'Geometric Average',
                                 'Exponential Smoothing', 'Median-Based', 'Trimmed Mean', 'Harmonic Mean']
            selected_method = st.selectbox("Select an averaging method:", averaging_methods)
            pivot_atafs = prepare_pivot_atafs(df)
            
            if selected_method == 'Simple Moving Average':
                pivot_atafs=visualize_sma(pivot_atafs)
            elif selected_method == 'Volume Weighted Average':
                pivot_atafs=visualize_vwa(pivot_atafs, df)
            elif selected_method == 'Geometric Average':
                pivot_atafs=visualize_ga(pivot_atafs)          
            elif selected_method == 'Exponential Smoothing':
                pivot_atafs=visualize_esa(pivot_atafs)
            elif selected_method == 'Median-Based':
                pivot_atafs=visualize_median(pivot_atafs)
            elif selected_method == 'Trimmed Mean':
                pivot_atafs=visualize_trimmed(pivot_atafs)
            elif selected_method == 'Harmonic Mean':
                pivot_atafs=visualize_harmonic(pivot_atafs)

        elif option == 'LDFs Summary Report':
            st.write("We have 3 different visualization graphs to help you analyze the Average LDFs better. Please select one from the dropdown below:")
            summary_options = ['Select a visualization method for the LDF Summary Report','Heatmap of LDF Averaging Methods', 'Radar Chart', 'Boxplot Charts']
            selected_summary_option = st.selectbox("Choose a visualization:", summary_options)
            triangle = prepare_triangle(df)
            # summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
            pivot_atafs = prepare_pivot_atafs(df)
            pivot_atafs=calc_sma(pivot_atafs)
            pivot_atafs=calc_vwa(pivot_atafs, df)
            pivot_atafs=calc_ga(pivot_atafs)
            pivot_atafs=calc_esa(pivot_atafs)
            pivot_atafs=calc_median(pivot_atafs)
            pivot_atafs=calc_trimmed(pivot_atafs)
            pivot_atafs=calc_harmonic(pivot_atafs)
            summary_IBNR_df = create_summary_table(pivot_atafs, df)

            if selected_summary_option == 'Heatmap of LDF Averaging Methods':
                visualize_heatmap(summary_IBNR_df)
            elif selected_summary_option == 'Radar Chart':
                year = st.selectbox("Select Accident Year:", range(1995, 2016))
                visualize_radar_chart(summary_IBNR_df, year)
            elif selected_summary_option == 'Boxplot Charts':
                visualize_boxplot(summary_IBNR_df)

        elif option == 'Actuary Methods for IBNR':
            st.write("Great! You have finally reached the Actuary Method implementation!")
            methods = ['Select an Actuary Method','Chain-Ladder Method', 'Bornhuetter-Ferguson (BF) Method', 'Loss Ratio Method']
            selected_method = st.selectbox("Choose an actuary method from the dropdown menu below:", methods)
            triangle = prepare_triangle(df)

            if selected_method == 'Chain-Ladder Method':
                visualize_chain_ladder(triangle)
            elif selected_method == 'Bornhuetter-Ferguson (BF) Method':
                visualize_bf_method(triangle)
            elif selected_method == 'Loss Ratio Method':
                visualize_loss_ratio(df)

        elif option == 'Ultimate Claims & IBNR Reserves Summary Report':
            st.write("Let's analyze the comparisons between the outputs of each actuarial method! Please select an option from the dropdown below:")
            summary_options = ['Choose one option','Ultimate Claims Summary', 'IBNR Reserves Summary']
            selected_summary_option = st.selectbox("Select Summary Type:", summary_options)
            triangle = prepare_triangle(df)
            summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)

            if selected_summary_option == 'Ultimate Claims Summary':
                visualize_ultimate_claims_summary(summary_ultimate_claims_df)
            elif selected_summary_option == 'IBNR Reserves Summary':
                visualize_ibnr_reserves_summary(summary_IBNR_df)

        elif option == 'Diagnostic Metrics Evaluation':
            st.write("Let's analyze a few diagnostic metrics to analyze the dataset even better! Please select an option from the dropdown below:")
            diagnostic_options = ['Choose one option','Paid to Reported Incurred Claims Ratio', 'Incurred to Earned Premium Ratio Heatmap']
            selected_diagnostic_option = st.selectbox("Select Diagnostic Metric:", diagnostic_options)
            paid_to_reported_triangle, incurred_to_earned_triangle = prepare_diagnostic_metrics(df)

            if selected_diagnostic_option == 'Paid to Reported Incurred Claims Ratio':
                plot_heatmap_triangle(paid_to_reported_triangle, "Paid to Reported Incurred Claims Ratio Heatmap")
            elif selected_diagnostic_option == 'Incurred to Earned Premium Ratio Heatmap':
                plot_heatmap_triangle(incurred_to_earned_triangle, "Incurred to Earned Premium Ratio Heatmap")
    else:
        st.error("Please upload and process the dataset first!")


