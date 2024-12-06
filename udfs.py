import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import trim_mean
from math import pi
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import ListedColormap
from matplotlib import colors as mcolors
import streamlit as st

# Function to calculate derived columns
def calculate_derived_columns(df):
    # Calculate inflation factors and inflated claims
    def calculate_inflation_factor(accident_year, min_factor=1.1, max_factor=1.6):
        year_range = df['accident_year'].max() - df['accident_year'].min()
        relative_position = (accident_year - df['accident_year'].min()) / year_range
        return min_factor + relative_position * (max_factor - min_factor)
    
    df['inflation_factor'] = df['accident_year'].apply(calculate_inflation_factor)
    df['reported_claims_inflated'] = df['incurred'] * df['inflation_factor']
    df['paid_claims_inflated'] = df['cumpaid'] * df['inflation_factor']

    # Calculate ratios and incremental values
    base_year = df['accident_year'].min()
    df['inflation_ratio'] = df['accident_year'].apply(
        lambda x: calculate_inflation_factor(x) / calculate_inflation_factor(base_year))
    df['incremental_reported'] = df.groupby('accident_year')['incurred'].diff().fillna(df['incurred'])
    df['incremental_paid'] = df.groupby('accident_year')['cumpaid'].diff().fillna(df['cumpaid'])
    epsilon = 1e-6
    df['expected_loss_ratio'] = df['incurred'] / (df['earned_premium'] + epsilon)
    df['ultimate_claims'] = df['incurred'] + df['case_reserves']
    df['paid_to_reported_ratio'] = df['cumpaid'] / (df['incurred'] + epsilon)
    df['paid_to_reported_ratio_inflated'] = df['cumpaid'] / (df['reported_claims_inflated'] + epsilon)
    df['incurred_to_earned_ratio'] = df['incurred'] / (df['earned_premium'] + epsilon)
    return df

# Define the function to calculate and plot ATAFs
def calculate_and_plot_ataf(df):
    df['next_year_incurred'] = df.groupby('accident_year')['incurred'].shift(-1)
    df['ATAF'] = df['next_year_incurred'] / df['incurred']
    df.drop(columns=['next_year_incurred'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN
    
    heatmap_data = df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    norm = mcolors.Normalize(vmin=heatmap_data.min().min(), vmax=heatmap_data.max().max())

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap=cmap, norm=norm,
                cbar_kws={'label': 'Age-to-Age Factor'}, linewidths=.5, linecolor='black')
    plt.title('Heatmap of Age-to-Age Factors (ATAFs) with Coolwarm Color Scaling')
    plt.ylabel('Accident Year')
    plt.xlabel('Development Year')
    plt.tight_layout()  # Adjust layout to not cut off labels
    st.pyplot(plt)

def calculate_ataf(df):
    df['next_year_incurred'] = df.groupby('accident_year')['incurred'].shift(-1)
    df['ATAF'] = df['next_year_incurred'] / df['incurred']
    df.drop(columns=['next_year_incurred'], inplace=True)
    df['ATAF'] = df['ATAF'].replace([np.inf, -np.inf], np.nan)  # Replacing infinite values with NaN

def prepare_pivot_atafs(df):
    return df.pivot_table(values='ATAF', index='accident_year', columns='development_year', aggfunc='mean')

def visualize_sma(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'SMA_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(min(len(row), span)).mean(), axis=1)
    print("***********************************************")
    print(pivot_atafs)
    plt.figure(figsize=(10, 5))
    pivot_atafs['SMA_10'].plot(title='Simple Moving Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    sma_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['SMA_10'],
        'Latest 5 Years': pivot_atafs['SMA_5'],
        'Latest 3 Years': pivot_atafs['SMA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(sma_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs
    
def visualize_vwa(pivot_atafs, df):
    # Create weights pivot table using incurred claims
    weights = df.pivot_table(values='incurred', index='accident_year', columns='development_year', aggfunc='sum')
    
    # Ensure only valid development years are considered
    valid_dev_years = [col for col in pivot_atafs.columns if col in weights.columns]
    pivot_atafs_valid = pivot_atafs[valid_dev_years]
    weights_valid = weights[valid_dev_years]

    # Define a function to calculate VWA for a given span
    def volume_weighted_average(series, weights, span):
        valid_data = series.dropna().tail(min(len(series), span))
        valid_weights = weights.loc[valid_data.index].tail(len(valid_data))
        if valid_data.empty or valid_weights.sum() == 0:
            return np.nan
        return (valid_data * valid_weights).sum() / valid_weights.sum()

    # Calculate VWA LDFs for various spans
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'VWA_{span}'] = pivot_atafs_valid.apply(
            lambda row: volume_weighted_average(row, weights_valid.loc[row.name], span), axis=1)

    # Plot the VWA for the latest 10 years
    plt.figure(figsize=(10, 5))
    pivot_atafs['VWA_10'].plot(title='Volume Weighted Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Prepare and display a comparative table for VWAs
    vwa_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['VWA_10'],
        'Latest 5 Years': pivot_atafs['VWA_5'],
        'Latest 3 Years': pivot_atafs['VWA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(vwa_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs
    
def geometric_average(series, span):
    valid_data = series.dropna().tail(min(len(series), span))
    if valid_data.empty:
        return np.nan
    # Calculate the geometric mean
    return np.prod(valid_data)**(1.0 / len(valid_data))

def visualize_ga(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'GA_{span}'] = pivot_atafs.apply(lambda row: geometric_average(row, span), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['GA_10'].plot(title='Geometric Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    ga_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['GA_10'],
        'Latest 5 Years': pivot_atafs['GA_5'],
        'Latest 3 Years': pivot_atafs['GA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(ga_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs

def visualize_esa(pivot_atafs):
    alpha = 0.3
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'ESA_{span}'] = pivot_atafs.apply(
            lambda row: row.dropna().tail(span).ewm(alpha=alpha).mean().iloc[-1] if len(row.dropna().tail(span)) > 0 else np.nan,
            axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['ESA_10'].plot(title='Exponential Smoothing Average LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    esa_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['ESA_10'],
        'Latest 5 Years': pivot_atafs['ESA_5'],
        'Latest 3 Years': pivot_atafs['ESA_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(esa_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs
    
def visualize_median(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Median_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(span).median(), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['Median_10'].plot(title='Median-Based LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    median_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['Median_10'],
        'Latest 5 Years': pivot_atafs['Median_5'],
        'Latest 3 Years': pivot_atafs['Median_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(median_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs
    
def visualize_trimmed(pivot_atafs):
    def trimmed_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return trim_mean(valid_data, proportiontocut=0.1) if len(valid_data) > 0 else np.nan
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Trimmed_{span}'] = pivot_atafs.apply(lambda row: trimmed_mean_ldf(row, span), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['Trimmed_10'].plot(title='Trimmed Mean LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    trimmed_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['Trimmed_10'],
        'Latest 5 Years': pivot_atafs['Trimmed_5'],
        'Latest 3 Years': pivot_atafs['Trimmed_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(trimmed_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs
    
def visualize_harmonic(pivot_atafs):
    def harmonic_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return len(valid_data) / np.sum(1.0 / valid_data) if len(valid_data) > 0 and all(valid_data > 0) else np.nan
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Harmonic_{span}'] = pivot_atafs.apply(lambda row: harmonic_mean_ldf(row, span), axis=1)
    plt.figure(figsize=(10, 5))
    pivot_atafs['Harmonic_10'].plot(title='Harmonic Mean LDFs for the Latest 10 Years', marker='o', linestyle='-')
    plt.ylabel('LDF')
    plt.xlabel('Accident Year')
    plt.grid(True)
    st.pyplot(plt)

    # Comparative table
    harmonic_table = pd.DataFrame({
        'Latest 10 Years': pivot_atafs['Harmonic_10'],
        'Latest 5 Years': pivot_atafs['Harmonic_5'],
        'Latest 3 Years': pivot_atafs['Harmonic_3']
    }).T
    st.write("### Comparison between LDFs for Latest 10, 5, and 3 Years")
    st.dataframe(harmonic_table.style.background_gradient(cmap='YlGnBu', axis=1).format("{:.2f}"))
    return pivot_atafs
    
def calc_sma(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'SMA_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(min(len(row), span)).mean(), axis=1)
    return pivot_atafs
    
def calc_vwa(pivot_atafs, df):
    # Create weights pivot table using incurred claims
    weights = df.pivot_table(values='incurred', index='accident_year', columns='development_year', aggfunc='sum')
    
    # Ensure only valid development years are considered
    valid_dev_years = [col for col in pivot_atafs.columns if col in weights.columns]
    pivot_atafs_valid = pivot_atafs[valid_dev_years]
    weights_valid = weights[valid_dev_years]

    # Define a function to calculate VWA for a given span
    def volume_weighted_average(series, weights, span):
        valid_data = series.dropna().tail(min(len(series), span))
        valid_weights = weights.loc[valid_data.index].tail(len(valid_data))
        if valid_data.empty or valid_weights.sum() == 0:
            return np.nan
        return (valid_data * valid_weights).sum() / valid_weights.sum()

    # Calculate VWA LDFs for various spans
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'VWA_{span}'] = pivot_atafs_valid.apply(
            lambda row: volume_weighted_average(row, weights_valid.loc[row.name], span), axis=1)
    return pivot_atafs
    
def calc_ga(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'GA_{span}'] = pivot_atafs.apply(lambda row: geometric_average(row, span), axis=1)
    return pivot_atafs

def calc_esa(pivot_atafs):
    alpha = 0.3
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'ESA_{span}'] = pivot_atafs.apply(
            lambda row: row.dropna().tail(span).ewm(alpha=alpha).mean().iloc[-1] if len(row.dropna().tail(span)) > 0 else np.nan,
            axis=1)
    return pivot_atafs
    
def calc_median(pivot_atafs):
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Median_{span}'] = pivot_atafs.apply(lambda row: row.dropna().tail(span).median(), axis=1)
    return pivot_atafs
    
def calc_trimmed(pivot_atafs):
    def trimmed_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return trim_mean(valid_data, proportiontocut=0.1) if len(valid_data) > 0 else np.nan
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Trimmed_{span}'] = pivot_atafs.apply(lambda row: trimmed_mean_ldf(row, span), axis=1)
    return pivot_atafs
    
def calc_harmonic(pivot_atafs):
    def harmonic_mean_ldf(series, span):
        valid_data = series.dropna().tail(span)
        return len(valid_data) / np.sum(1.0 / valid_data) if len(valid_data) > 0 and all(valid_data > 0) else np.nan
    spans = [10, 5, 3]
    for span in spans:
        pivot_atafs[f'Harmonic_{span}'] = pivot_atafs.apply(lambda row: harmonic_mean_ldf(row, span), axis=1)
    return pivot_atafs

def create_summary_table(pivot_atafs,df):
    return pd.DataFrame({
        'Simple Mean (10)': pivot_atafs['SMA_10'],
        'Volume Weighted (10)': pivot_atafs['VWA_10'],
        'Geometric Mean (10)': pivot_atafs['GA_10'],
        'Exponential Smoothing (10)': pivot_atafs['ESA_10'],
        'Median (10)': pivot_atafs['Median_10'],
        'Trimmed Mean (10)': pivot_atafs['Trimmed_10'],
        'Harmonic Mean (10)': pivot_atafs['Harmonic_10']
    })

def visualize_heatmap(summary_table):
    plt.figure(figsize=(12, 8))
    sns.heatmap(summary_table, annot=True, fmt=".2f", cmap="coolwarm", linewidths=.5)
    plt.title('Heatmap of LDF Averaging Methods (Latest 10 Years)')
    plt.ylabel('Accident Year')
    plt.xlabel('LDF Methods')
    st.pyplot(plt)
  
def visualize_radar_chart(summary_table, year):
    methods = summary_table.columns
    angles = [n / float(len(methods)) * 2 * pi for n in range(len(methods))]
    angles += angles[:1]  # Complete the circle

    values = summary_table.loc[year].tolist()
    values += values[:1]  # Complete the circle

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='blue', linewidth=2)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_title(f'LDF Comparison Across Methods for Accident Year {year}')
    st.pyplot(fig)

def visualize_boxplot(summary_table):
    melted_summary = summary_table.melt(var_name='Method', value_name='LDF')

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=melted_summary, x='Method', y='LDF')
    plt.xticks(rotation=45)
    plt.title('Distribution of LDFs Across Averaging Methods')
    plt.ylabel('LDF')
    st.pyplot(plt)

def prepare_triangle(df):
    return df.pivot(index='accident_year', columns='development_year', values='incurred')

# Chain-Ladder Method
def visualize_chain_ladder(triangle):
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    ibnr_reserve = projected_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)

    def format_number(x):
        if x >= 1e6:
            return f'{x/1e6:.1f}M'  # Millions
        elif x >= 1e3:
            return f'{x/1e3:.1f}K'  # Thousands
        else:
            return f'{x:.1f}'  # Keep as is if less than 1000

    # Define the Coolwarm color palette
    cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)
    
    # Normalize the color map to fit the range of data
    norm = mcolors.Normalize(vmin=projected_triangle.min().min(), vmax=projected_triangle.max().max())

    formatted_triangle = projected_triangle.map(format_number)

    # Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(projected_triangle, cmap='coolwarm', annot=formatted_triangle, norm=norm, fmt="", linewidths=0.5, linecolor='black',annot_kws={'size': 6})
    plt.title('Projected Incurred Claim Amount Triangle')
    plt.ylabel('Accident Year')
    plt.xlabel('Development Year')
    st.pyplot(plt)

    # Bar chart for IBNR reserves
    plt.figure(figsize=(10, 6))
    sns.barplot(x=ibnr_reserve.index, y=ibnr_reserve.values, palette="Blues_d",hue=ibnr_reserve.values)
    # plt.grid(True, linestyle='--', linewidth=0.5)
    plt.title("IBNR Reserves by Accident Year")
    plt.xlabel("Accident Year")
    plt.ylabel("IBNR Reserve Amount")
    st.pyplot(plt)

    # Line chart for ultimate claim amount
    ultimate_claim_amount = projected_triangle.sum(axis=1)
    plt.figure(figsize=(10, 6))
    plt.plot(ultimate_claim_amount.index, ultimate_claim_amount.values, marker='o', linestyle='-', color='teal')
    plt.title("Ultimate Claims Amount by Accident Year (Chain Ladder Method)")
    plt.xlabel("Accident Year")
    plt.ylabel("Ultimate Claim Amount")
    st.pyplot(plt)

# Bornhuetter-Ferguson (BF) Method
def visualize_bf_method(triangle):
    development_factors = []
    for col in range(triangle.shape[1] - 1):
        current_sum = triangle.iloc[:, col].sum()
        next_sum = triangle.iloc[:, col + 1].sum()
        factor = next_sum / current_sum if current_sum != 0 else 1
        development_factors.append(factor)

    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                previous_value = projected_triangle.iloc[row, col - 1]
                if not pd.isna(previous_value):
                    projected_triangle.iloc[row, col] = previous_value * development_factors[col - 1]

    bf_ultimate_claims = projected_triangle.sum(axis=1)
    bf_reserves = (1 - (triangle.sum(axis=1) / bf_ultimate_claims)) * bf_ultimate_claims

    # Bar chart for BF reserves
    plt.figure(figsize=(10, 6))
    sns.barplot(x=bf_reserves.index, y=bf_reserves.values, palette="Blues_d", hue=bf_reserves.values)
    plt.title("BF Reserves by Accident Year")
    plt.xlabel("Accident Year")
    plt.ylabel("BF Reserve Amount")
    st.pyplot(plt)

    # Bar chart for ultimate claims
    plt.figure(figsize=(10, 6))
    sns.barplot(x=bf_ultimate_claims.index, y=bf_ultimate_claims.values, palette="Greens_d",hue=bf_ultimate_claims.values)
    plt.title("Ultimate Claims Amount by Accident Year (BF Method)")
    plt.xlabel("Accident Year")
    plt.ylabel("Ultimate Claims Amount")
    st.pyplot(plt)

    incurred_claims = triangle.sum(axis=1)  # Sum of incurred claims for each accident year

    # Prepare the data for stacked bar plot
    data_for_comparison = pd.DataFrame({
        'Incurred Claims': incurred_claims,
        'BF Reserves': bf_reserves
    })
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    data_for_comparison.plot(kind='bar', stacked=True, ax=ax, color=['lightblue', 'salmon'])
    ax.set_title("Incurred Claims vs BF Reserves by Accident Year", fontsize=16)
    ax.set_xlabel("Accident Year", fontsize=12)
    ax.set_ylabel("Amount", fontsize=12)
    plt.tight_layout()
    
    # Display the plot in Streamlit
    st.pyplot(fig)

# Loss Ratio Method
def visualize_loss_ratio(df):
    triangle = prepare_triangle(df)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')

    incurred_claims = triangle.sum(axis=1)
    total_earned_premiums = earned_premiums.sum(axis=1)
    loss_ratios = incurred_claims / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - incurred_claims

    # Bar chart for loss ratios
    plt.figure(figsize=(12, 7))
    sns.barplot(x=loss_ratios.index, y=loss_ratios.values, palette="Reds_d",hue=loss_ratios.values)
    plt.title("Loss Ratio by Accident Year")
    plt.xlabel("Accident Year")
    plt.ylabel("Loss Ratio")
    st.pyplot(plt)

    # Bar chart for expected claims
    plt.figure(figsize=(10, 6))
    sns.barplot(x=expected_claims.index, y=expected_claims.values, palette="Purples_d",hue=expected_claims.values)
    plt.title("Ultimate Claims Amount by Accident Year (Loss Ratio Method)")
    plt.xlabel("Accident Year")
    plt.ylabel("Expected Claims Amount")
    st.pyplot(plt)

    # Stacked bar chart for incurred claims vs reserves
    data_for_comparison = pd.DataFrame({
        'Incurred Claims': incurred_claims,
        'Loss Ratio Reserves': loss_ratio_reserves
    })
    data_for_comparison.plot(kind='bar', stacked=True, figsize=(10, 6), color=['lightcoral', 'gold'])
    plt.title("Incurred Claims vs Loss Ratio Reserves by Accident Year")
    plt.xlabel("Accident Year")
    plt.ylabel("Amount")
    st.pyplot(plt)

def prepare_summary_dataframes(triangle, df):
    # Calculate required components
    development_factors = [triangle.iloc[:, col + 1].sum() / triangle.iloc[:, col].sum() 
                           for col in range(triangle.shape[1] - 1)]
    projected_triangle = triangle.copy()
    for col in range(1, projected_triangle.shape[1]):
        for row in range(projected_triangle.shape[0]):
            if pd.isna(projected_triangle.iloc[row, col]):
                projected_triangle.iloc[row, col] = projected_triangle.iloc[row, col - 1] * development_factors[col - 1]

    ibnr_reserve = projected_triangle.sum(axis=1) - triangle.sum(axis=1, min_count=1)
    bf_reserves = (1 - (triangle.sum(axis=1) / projected_triangle.sum(axis=1))) * projected_triangle.sum(axis=1)
    earned_premiums = df.pivot(index='accident_year', columns='development_year', values='earned_premium')
    total_earned_premiums = earned_premiums.sum(axis=1)
    loss_ratios = triangle.sum(axis=1) / total_earned_premiums
    expected_claims = total_earned_premiums * loss_ratios
    loss_ratio_reserves = expected_claims - triangle.sum(axis=1)

    # Prepare summary DataFrames
    summary_IBNR_df = pd.DataFrame({
        "Accident Year": projected_triangle.index,
        "IBNR Reserves (Chain Ladder)": ibnr_reserve.values,
        "IBNR Reserves (BF Method)": bf_reserves.values,
        "IBNR Reserves (Loss Ratio)": loss_ratio_reserves.values
    })

    summary_ultimate_claims_df = pd.DataFrame({
        "Accident Year": projected_triangle.index,
        "Ultimate Claims (Chain Ladder)": projected_triangle.sum(axis=1).values,
        "Ultimate Claims (BF Method)": projected_triangle.sum(axis=1).values,  # Replace if BF differs
        "Ultimate Claims (Loss Ratio)": expected_claims.values
    })

    return summary_IBNR_df, summary_ultimate_claims_df

def visualize_ibnr_reserves_summary(summary_IBNR_df):
    plt.figure(figsize=(14, 7))
    bar_width = 0.25
    x = np.arange(len(summary_IBNR_df["Accident Year"]))
    plt.bar(x - bar_width, summary_IBNR_df["IBNR Reserves (Chain Ladder)"], bar_width, label="Chain Ladder", color="#1f77b4")
    plt.bar(x, summary_IBNR_df["IBNR Reserves (BF Method)"], bar_width, label="BF Method", color="#ff7f0e")
    plt.bar(x + bar_width, summary_IBNR_df["IBNR Reserves (Loss Ratio)"], bar_width, label="Loss Ratio", color="#2ca02c")
    plt.title("Comparison of IBNR Reserves by Method", fontsize=16)
    plt.xlabel("Accident Year", fontsize=12)
    plt.ylabel("IBNR Reserves", fontsize=12)
    plt.xticks(x, summary_IBNR_df["Accident Year"], rotation=45, fontsize=10)
    plt.legend(title="Method", fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)

    st.write("### IBNR Reserves Summary Table")
    st.dataframe(summary_IBNR_df.style.format("{:.2f}").set_table_styles(
        [{'selector': 'th', 'props': [('font-size', '12pt')]}]))

def visualize_ultimate_claims_summary(summary_ultimate_claims_df):
    plt.figure(figsize=(15, 7))
    bar_width = 0.25
    x = np.arange(len(summary_ultimate_claims_df["Accident Year"]))
    plt.bar(x - bar_width, summary_ultimate_claims_df["Ultimate Claims (Chain Ladder)"], bar_width, label="Chain Ladder", color="#1f77b4")
    plt.bar(x, summary_ultimate_claims_df["Ultimate Claims (BF Method)"], bar_width, label="BF Method", color="#ff7f0e")
    plt.bar(x + bar_width, summary_ultimate_claims_df["Ultimate Claims (Loss Ratio)"], bar_width, label="Loss Ratio", color="#2ca02c")
    plt.title("Comparison of Ultimate Claims by Method", fontsize=16)
    plt.xlabel("Accident Year", fontsize=12)
    plt.ylabel("Ultimate Claims", fontsize=12)
    plt.xticks(x, summary_ultimate_claims_df["Accident Year"], rotation=45, fontsize=10)
    plt.legend(title="Method", fontsize=10, title_fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    st.pyplot(plt)

    st.write("### Ultimate Claims Summary Table")
    st.dataframe(summary_ultimate_claims_df.style.format("{:.2f}").set_table_styles(
        [{'selector': 'th', 'props': [('font-size', '12pt')]}]))

def prepare_diagnostic_metrics(df):
    # Pivot data into triangles
    paid_to_reported_triangle = df.pivot(index='accident_year', columns='development_year', values='paid_to_reported_ratio')
    incurred_to_earned_triangle = df.pivot(index='accident_year', columns='development_year', values='incurred_to_earned_ratio')
    return paid_to_reported_triangle, incurred_to_earned_triangle

def plot_heatmap_triangle(data, title, cmap='coolwarm'):
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        data,
        annot=True,  # Display values on the heatmap
        fmt=".2f",  # Limit decimal places
        cmap=cmap,
        linewidths=0.5,  # Add gridlines
        cbar_kws={"label": "Ratio"}  # Label for the color bar
    )
    plt.title(title, fontsize=16)
    plt.xlabel("Development Year", fontsize=12)
    plt.ylabel("Accident Year", fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    st.pyplot(plt)
