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
load_dotenv()

# Title of the app
st.title('IBNR Loss Reserves Usecase')

# Sidebar navigation
st.sidebar.title("Tool Capabilities")
option = st.sidebar.radio('Choose a section:', ['Input Dataset', 'Average LDFs', 
                                                'LDFs Summary Report', 'Actuary Methods for IBNR',
                                                'Diagnostic Metrics Evaluation',
                                                'Ultimate Claims & IBNR Reserves Summary Report'])

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

            heatmap_methods = ['Select an option','Show Age-to-Age Factors Heatmap for Incurred Claim Amount','Show Age-to-Age Factors Heatmap for Paid Claim Amount']
            selected_method = st.selectbox("Select an ATAF Heatmap chart", heatmap_methods)

            if selected_method == 'Show Age-to-Age Factors Heatmap for Incurred Claim Amount':
                with st.spinner('Generating heatmap...'):
                    calculate_and_plot_ataf(st.session_state.uploaded_df)

            elif selected_method == 'Show Age-to-Age Factors Heatmap for Paid Claim Amount':
                with st.spinner('Generating heatmap...'):
                    calculate_and_plot_ataf_cumpaid(st.session_state.uploaded_df)
            # if st.button('Show Age-to-Age Factors Heatmap'):
            #     with st.spinner('Generating heatmap...'):
            #         calculate_and_plot_ataf(st.session_state.uploaded_df)
        except Exception as e:
            st.error(f"Failed to process file: {e}")

else:
    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        # calculate_ataf(st.session_state.uploaded_df)
        # calculate_ataf_cumpaid(st.session_state.uploaded_df)

        if option == 'Average LDFs':
            st.write("We have 7 averaging methods available to calculate LDFs from the ATAF. Please select one from the dropdown below:")

            claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)

            if claim_method == 'Incurred Claim Amount':
                averaging_methods = ['Select an averaging method','Simple Moving Average', 'Volume Weighted Average', 'Exponential Smoothing', 'Median-Based', 'Trimmed Mean']
                selected_method = st.selectbox("Select an averaging method:", averaging_methods)
                df = st.session_state.uploaded_df
                calculate_ataf(st.session_state.uploaded_df)
                pivot_atafs = prepare_pivot_atafs(df)
                
                if selected_method == 'Simple Moving Average':
                    st.write(":green-background[:bulb: Hey! Simple Moving Average is best used when the ATAF data points are consistent and there are no significant outliers. It works best for scenarios where data variability is relatively low.]")
                    pivot_atafs=visualize_sma(pivot_atafs)
                elif selected_method == 'Volume Weighted Average':
                    st.write(":green-background[:bulb: Hey! We have incorporated the Incurred Claims Amount as weights while calculating the volume weighted average. This method is particularly very useful in scenarios where claim amounts are highly variable. By weighting more recent claims or those with larger impacts (such as higher claim amounts) more heavily, it can provide a more accurate reflection of recent trends and expected future developments.]")
                    pivot_atafs=visualize_vwa(pivot_atafs, df)
                # elif selected_method == 'Geometric Average':
                #     pivot_atafs=visualize_ga(pivot_atafs)          
                elif selected_method == 'Exponential Smoothing':
                    st.write(":green-background[:bulb: Hey! We have applied an ideal Exponential Smoothing factor for this calculation. This method is particularly useful when we want to smooth out short-term fluctuations and highlight long-term trends or cycles.]")
                    pivot_atafs=visualize_esa(pivot_atafs)
                elif selected_method == 'Median-Based':
                    st.write(":green-background[:bulb: Hey! Median Based Average is particularly robust against outliers, which is beneficial in handling claims data that often contains large, atypical payouts. It's best used for skewed distributions or when outliers are present and could distort the mean.]")
                    pivot_atafs=visualize_median(pivot_atafs)
                elif selected_method == 'Trimmed Mean':
                    st.write(":green-background[:bulb: Hey! Trimmed Mean Average particularly strikes a balance between excluding outliers and retaining data. It's useful when the data set includes some extreme claims that should not overly influence the LDF but also contains valuable information in the outer ranges of data distribution.]")
                    pivot_atafs=visualize_trimmed(pivot_atafs)
                # elif selected_method == 'Harmonic Mean':
                #     pivot_atafs=visualize_harmonic(pivot_atafs)
                    
            elif claim_method == 'Paid Claim Amount':
                averaging_methods = ['Select an averaging method','Simple Moving Average', 'Volume Weighted Average', 'Exponential Smoothing', 'Median-Based', 'Trimmed Mean']
                selected_method = st.selectbox("Select an averaging method:", averaging_methods)
                df = st.session_state.uploaded_df
                calculate_ataf_cumpaid(st.session_state.uploaded_df)
                pivot_claim_atafs = prepare_pivot_claim_atafs(df)
                
                if selected_method == 'Simple Moving Average':
                    st.write(":green-background[:bulb: Hey! Simple Moving Average is best used when the ATAF data points are consistent and there are no significant outliers. It works best for scenarios where data variability is relatively low.]")
                    pivot_atafs=visualize_sma(pivot_claim_atafs)
                elif selected_method == 'Volume Weighted Average':
                    st.write(":green-background[:bulb: Hey! We have incorporated the Paid Claims Amount as weights while calculating the volume weighted average. This method is particularly very useful in scenarios where claim amounts are highly variable. By weighting more recent claims or those with larger impacts (such as higher claim amounts) more heavily, it can provide a more accurate reflection of recent trends and expected future developments.]")
                    pivot_atafs=visualize_vwa(pivot_claim_atafs, df)
                # elif selected_method == 'Geometric Average':
                #     pivot_atafs=visualize_ga(pivot_claim_atafs)          
                elif selected_method == 'Exponential Smoothing':
                    st.write(":green-background[:bulb: Hey! We have applied an ideal Exponential Smoothing factor for this calculation. This method is particularly useful when we want to smooth out short-term fluctuations and highlight long-term trends or cycles.]")
                    pivot_atafs=visualize_esa(pivot_claim_atafs)
                elif selected_method == 'Median-Based':
                    st.write(":green-background[:bulb: Hey! Median Based Average is particularly robust against outliers, which is beneficial in handling claims data that often contains large, atypical payouts. It's best used for skewed distributions or when outliers are present and could distort the mean.]")
                    pivot_atafs=visualize_median(pivot_claim_atafs)
                elif selected_method == 'Trimmed Mean':
                    st.write(":green-background[:bulb: Hey! Trimmed Mean Average particularly strikes a balance between excluding outliers and retaining data. It's useful when the data set includes some extreme claims that should not overly influence the LDF but also contains valuable information in the outer ranges of data distribution.]")
                    pivot_atafs=visualize_trimmed(pivot_claim_atafs)
                # elif selected_method == 'Harmonic Mean':
                #     pivot_atafs=visualize_harmonic(pivot_claim_atafs)        
                    
        elif option == 'LDFs Summary Report':
            st.write("We have 3 different visualization graphs to help you analyze the Average LDFs better. Please select one from the dropdown below:")
            
            claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)

            if claim_method == 'Incurred Claim Amount':
                
                summary_options = ['Select a visualization method for the LDF Summary Report','Heatmap of LDF Averaging Methods', 'Radar Chart', 'Boxplot Charts']
                selected_summary_option = st.selectbox("Choose a visualization:", summary_options)
                df = st.session_state.uploaded_df
                calculate_ataf(st.session_state.uploaded_df)
                triangle = prepare_triangle(df)
                # summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
                pivot_atafs = prepare_pivot_atafs(df)
                pivot_atafs=calc_sma(pivot_atafs)
                pivot_atafs=calc_vwa(pivot_atafs, df)
                # pivot_atafs=calc_ga(pivot_atafs)
                pivot_atafs=calc_esa(pivot_atafs)
                pivot_atafs=calc_median(pivot_atafs)
                pivot_atafs=calc_trimmed(pivot_atafs)
                # pivot_atafs=calc_harmonic(pivot_atafs)
                summary_IBNR_df = create_summary_table(pivot_atafs, df)
    
                if selected_summary_option == 'Heatmap of LDF Averaging Methods':
                    visualize_heatmap(summary_IBNR_df)
                elif selected_summary_option == 'Radar Chart':
                    year = st.selectbox("Select Accident Year:", range(1995, 2016))
                    visualize_radar_chart(summary_IBNR_df, year)
                elif selected_summary_option == 'Boxplot Charts':
                    visualize_boxplot(summary_IBNR_df)
                    
            elif claim_method == 'Paid Claim Amount':
                summary_options = ['Select a visualization method for the LDF Summary Report','Heatmap of LDF Averaging Methods', 'Radar Chart', 'Boxplot Charts']
                selected_summary_option = st.selectbox("Choose a visualization:", summary_options)
                df = st.session_state.uploaded_df
                calculate_ataf_cumpaid(st.session_state.uploaded_df)
                triangle = prepare_claim_triangle(df)
                # summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
                pivot_claim_atafs = prepare_pivot_claim_atafs(df)
                pivot_atafs=calc_sma(pivot_claim_atafs)
                pivot_atafs=calc_vwa(pivot_claim_atafs, df)
                # pivot_atafs=calc_ga(pivot_atafs)
                pivot_atafs=calc_esa(pivot_claim_atafs)
                pivot_atafs=calc_median(pivot_claim_atafs)
                pivot_atafs=calc_trimmed(pivot_claim_atafs)
                # pivot_atafs=calc_harmonic(pivot_atafs)
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

            claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)

            if claim_method == 'Incurred Claim Amount':
                methods = ['Select an Actuary Method','Chain-Ladder Method', 'Bornhuetter-Ferguson (BF) Method', 'Loss Ratio Method']
                selected_method = st.selectbox("Choose an actuary method from the dropdown menu below:", methods)
                df = st.session_state.uploaded_df
                calculate_ataf(st.session_state.uploaded_df)
                triangle = prepare_triangle(df)
    
                if selected_method == 'Chain-Ladder Method':
                    visualize_chain_ladder(triangle)
                elif selected_method == 'Bornhuetter-Ferguson (BF) Method':
                    visualize_bf_method(triangle)
                elif selected_method == 'Loss Ratio Method':
                    visualize_loss_ratio(df)

            elif claim_method == 'Paid Claim Amount':
                methods = ['Select an Actuary Method','Chain-Ladder Method', 'Bornhuetter-Ferguson (BF) Method', 'Loss Ratio Method']
                selected_method = st.selectbox("Choose an actuary method from the dropdown menu below:", methods)
                df = st.session_state.uploaded_df
                calculate_ataf_cumpaid(st.session_state.uploaded_df)
                triangle_claim = prepare_claim_triangle(df)
                if selected_method == 'Chain-Ladder Method':
                    visualize_chain_ladder(triangle_claim)
                elif selected_method == 'Bornhuetter-Ferguson (BF) Method':
                    visualize_bf_method(triangle_claim)
                elif selected_method == 'Loss Ratio Method':
                    visualize_claim_loss_ratio(df)                  

        elif option == 'Diagnostic Metrics Evaluation':
            st.write("Let's analyze a few diagnostic metrics to analyze the dataset even better! Please select an option from the dropdown below:")
            diagnostic_options = ['Choose one option','Paid to Reported Incurred Claims Ratio', 'Incurred to Earned Premium Ratio Heatmap']
            selected_diagnostic_option = st.selectbox("Select Diagnostic Metric:", diagnostic_options)
            paid_to_reported_triangle, incurred_to_earned_triangle = prepare_diagnostic_metrics(df)

            if selected_diagnostic_option == 'Paid to Reported Incurred Claims Ratio':
                plot_heatmap_triangle(paid_to_reported_triangle, "Paid Claims to Reported Incurred Claims Ratio Heatmap")
            elif selected_diagnostic_option == 'Incurred to Earned Premium Ratio Heatmap':
                plot_heatmap_triangle(incurred_to_earned_triangle, "Incurred Claims to Earned Premium Ratio Heatmap")

        elif option == 'Ultimate Claims & IBNR Reserves Summary Report':
            st.write("Let's analyze the comparisons between the outputs of each actuarial method! Please select an option from the dropdown below:")
            claim_amount_method = ['Choose one method','Incurred Claim Amount','Paid Claim Amount']
            claim_method = st.selectbox("Select Incurred or Paid Claim Amount:", claim_amount_method)
    
            if claim_method == 'Incurred Claim Amount':
                summary_options = ['Choose one option','Ultimate Claims Summary', 'IBNR Reserves Summary']
                selected_summary_option = st.selectbox("Select Summary Type:", summary_options)
                triangle = prepare_triangle(df)
                summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
        
                if selected_summary_option == 'Ultimate Claims Summary':
                    visualize_ultimate_claims_summary(summary_ultimate_claims_df)
                    
                    st.title("Let's now calculate the Final Ultimate Claims!")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final ultimate claims value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
                    chain_amount = chain_ladder(triangle)
                    bf_amount = bf_method(triangle)
                    loss_ratio_amount = loss_ratio(df)
                    # Title
                    st.write("Select the weight for Chain Ladder:")
                    # Slider configuration
                    chain_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Chain Ladder')
                    st.write("You selected", chain_value , "weight for Chain Ladder method")
                    
                    # Title
                    st.write("Select the weight for Bornhuetter-Ferguson (BF) Method:")
                    # Slider configuration
                    bf_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='BF')
                    st.write("You selected", bf_value , "weight for BF method")
                    
                    # Title
                    st.write("Select the weight for Loss Ratio Method:")
                    # Slider configuration
                    loss_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Loss Ratio')
                    st.write("You selected", loss_value , "weight for Loss Ratio method")

                    # Check the sum of the weights with a tolerance
                    total = chain_value + bf_value + loss_value
                    tolerance = 0.01  # Define a small tolerance, e.g., 0.01 or 0.001

                    if st.button('Show Final Ultimate Claim Amount'):
                        if abs(total - 1.0) <= tolerance:
                            chain_amount = chain_ladder(triangle)
                            bf_amount = bf_method(triangle)
                            loss_ratio_amount = loss_ratio(df)
                            
                            final_ultimate_claims = (chain_value * chain_amount +
                                                     bf_value * bf_amount +
                                                     loss_value * loss_ratio_amount)
                            st.write("Final Ultimate Incurred Claims Amount by Accident Year:")
                            final_ultimate_claims.index.names = ['Accident Year']
                            final_ultimate_claims.name = 'Final Ultimate Claims Amount'
                            st.dataframe(final_ultimate_claims)

                        else:
                            st.session_state['weights_correct'] = False
                            st.error(f"Please check if the weights add up to 1.0. Currently, they sum to {total:.2f}.")
                    
                elif selected_summary_option == 'IBNR Reserves Summary':
                    visualize_ibnr_reserves_summary(summary_IBNR_df)

                    st.title("Let's now calculate the Final IBNR Reserves!")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final IBNR reserves value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
                    chain_amount = chain_ladder_ibnr(triangle)
                    bf_amount = bf_method_ibnr(triangle)
                    loss_ratio_amount = loss_ratio_ibnr(df)
                    # Title
                    st.write("Select the weight for Chain Ladder:")
                    # Slider configuration
                    chain_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Chain Ladder')
                    st.write("You selected", chain_value , "weight for Chain Ladder method")
                    
                    # Title
                    st.write("Select the weight for Bornhuetter-Ferguson (BF) Method:")
                    # Slider configuration
                    bf_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='BF')
                    st.write("You selected", bf_value , "weight for BF method")
                    
                    # Title
                    st.write("Select the weight for Loss Ratio Method:")
                    # Slider configuration
                    loss_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Loss Ratio')
                    st.write("You selected", loss_value , "weight for Loss Ratio method")

                    # Check the sum of the weights with a tolerance
                    total = chain_value + bf_value + loss_value
                    tolerance = 0.01  # Define a small tolerance, e.g., 0.01 or 0.001

                    if st.button('Show Final IBNR Reserves Amount'):
                        if abs(total - 1.0) <= tolerance:
                            chain_amount = chain_ladder(triangle)
                            bf_amount = bf_method(triangle)
                            loss_ratio_amount = loss_ratio(df)
                            
                            final_ultimate_claims = (chain_value * chain_amount +
                                                     bf_value * bf_amount +
                                                     loss_value * loss_ratio_amount)
                            st.write("Final IBNR Reserves Amount by Accident Year:")
                            final_ultimate_claims.index.names = ['Accident Year']
                            final_ultimate_claims.name = 'Final IBNR Reserves Amount'
                            st.dataframe(final_ultimate_claims)

                        else:
                            st.session_state['weights_correct'] = False
                            st.error(f"Please check if the weights add up to 1.0. Currently, they sum to {total:.2f}.")
                        
            elif claim_method == 'Paid Claim Amount':
                summary_options = ['Choose one option','Ultimate Claims Summary', 'IBNR Reserves Summary']
                selected_summary_option = st.selectbox("Select Summary Type:", summary_options)
                triangle = prepare_claim_triangle(df)
                summary_IBNR_df, summary_ultimate_claims_df = prepare_summary_dataframes(triangle, df)
        
                if selected_summary_option == 'Ultimate Claims Summary':
                    visualize_ultimate_claims_summary(summary_ultimate_claims_df)
                    st.title("Let's now calculate the Final Ultimate Claims!")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final ultimate claims value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
                    chain_amount = chain_ladder(triangle)
                    bf_amount = bf_method(triangle)
                    loss_ratio_amount = loss_ratio(df)
                    # Title
                    st.write("Select the weight for Chain Ladder:")
                    # Slider configuration
                    chain_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Chain Ladder')
                    st.write("You selected", chain_value , "weight for Chain Ladder method")
                    
                    # Title
                    st.write("Select the weight for Bornhuetter-Ferguson (BF) Method:")
                    # Slider configuration
                    bf_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='BF')
                    st.write("You selected", bf_value , "weight for BF method")
                    
                    # Title
                    st.write("Select the weight for Loss Ratio Method:")
                    # Slider configuration
                    loss_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Loss Ratio')
                    st.write("You selected", loss_value , "weight for Loss Ratio method")

                    # Check the sum of the weights with a tolerance
                    total = chain_value + bf_value + loss_value
                    tolerance = 0.01  # Define a small tolerance, e.g., 0.01 or 0.001

                    if st.button('Show Final Ultimate Claim Amount'):
                        if abs(total - 1.0) <= tolerance:
                            chain_amount = chain_ladder(triangle)
                            bf_amount = bf_method(triangle)
                            loss_ratio_amount = loss_ratio(df)
                            
                            final_ultimate_claims = (chain_value * chain_amount +
                                                     bf_value * bf_amount +
                                                     loss_value * loss_ratio_amount)
                            st.write("Final Ultimate Incurred Claims Amount by Accident Year:")
                            final_ultimate_claims.index.names = ['Accident Year']
                            final_ultimate_claims.name = 'Final Ultimate Claims Amount'
                            st.dataframe(final_ultimate_claims)

                        else:
                            st.session_state['weights_correct'] = False
                            st.error(f"Please check if the weights add up to 1.0. Currently, they sum to {total:.2f}.")
                
                elif selected_summary_option == 'IBNR Reserves Summary':
                    visualize_ibnr_reserves_summary(summary_IBNR_df)  
                    
                    st.title("Let's now calculate the Final IBNR Reserves!")
                    st.write(":green-background[:bulb: We will use a combination of all three actuary methods to get the final IBNR reserves value. Please input weights for each actuary method. For example, Chain Ladder Method: 0.3, BF Method: 0.6, Loss Ratio Method: 0.1]")
                    chain_amount = chain_ladder_ibnr(triangle)
                    bf_amount = bf_method_ibnr(triangle)
                    loss_ratio_amount = loss_ratio_ibnr(df)
                    # Title
                    st.write("Select the weight for Chain Ladder:")
                    # Slider configuration
                    chain_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Chain Ladder')
                    st.write("You selected", chain_value , "weight for Chain Ladder method")
                    
                    # Title
                    st.write("Select the weight for Bornhuetter-Ferguson (BF) Method:")
                    # Slider configuration
                    bf_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='BF')
                    st.write("You selected", bf_value , "weight for BF method")
                    
                    # Title
                    st.write("Select the weight for Loss Ratio Method:")
                    # Slider configuration
                    loss_value = st.slider("Choose a value", min_value=0.0, max_value=1.0, step=0.1, value=0.5, key='Loss Ratio')
                    st.write("You selected", loss_value , "weight for Loss Ratio method")

                    # Check the sum of the weights with a tolerance
                    total = chain_value + bf_value + loss_value
                    tolerance = 0.01  # Define a small tolerance, e.g., 0.01 or 0.001

                    if st.button('Show Final IBNR Reserves Amount'):
                        if abs(total - 1.0) <= tolerance:
                            chain_amount = chain_ladder(triangle)
                            bf_amount = bf_method(triangle)
                            loss_ratio_amount = loss_ratio(df)
                            
                            final_ultimate_claims = (chain_value * chain_amount +
                                                     bf_value * bf_amount +
                                                     loss_value * loss_ratio_amount)
                            st.write("Final IBNR Reserves Amount by Accident Year:")
                            final_ultimate_claims.index.names = ['Accident Year']
                            final_ultimate_claims.name = 'Final IBNR Reserves Amount'
                            st.dataframe(final_ultimate_claims)

                        else:
                            st.session_state['weights_correct'] = False
                            st.error(f"Please check if the weights add up to 1.0. Currently, they sum to {total:.2f}.")
    
    else:
        st.error("Please upload and process the dataset first!")


