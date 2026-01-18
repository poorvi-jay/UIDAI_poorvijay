"""
Data_cleaning.py
----------------
Step A: Biometric Data Cleaning and Aggregation

This module:
1. Extracts Aadhaar biometric data from ZIP
2. Loads multiple CSV chunks
3. Cleans and standardizes data
4. Aggregates adult (17+) biometric updates at PINCODE–MONTH level
"""

import os
import zipfile
import pandas as pd


def load_and_clean_biometric_data(zip_path, extract_path):
    """
    Loads Aadhaar biometric update data and performs cleaning and aggregation.

    Parameters:
    zip_path (str): Path to biometric ZIP file
    extract_path (str): Folder to extract ZIP contents

    Returns:
    DataFrame: Aggregated biometric updates per pincode per month (17+ age group)
    """

    # Ensure extraction folder exists
    os.makedirs(extract_path, exist_ok=True)

    # Extract ZIP file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    # UIDAI data is inside a nested folder
    inner_path = os.path.join(extract_path, "api_data_aadhar_biometric")

    # Collect all CSV files
    csv_files = [
        os.path.join(inner_path, file)
        for file in os.listdir(inner_path)
        if file.endswith(".csv")
    ]

    # Read and merge all CSV chunks
    df_list = [pd.read_csv(file) for file in csv_files]
    df = pd.concat(df_list, ignore_index=True)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Remove rows without essential identifiers
    df = df.dropna(subset=['pincode', 'date'])

    # Missing biometric update counts imply zero updates
    df = df.fillna(0)

    # Convert date column to datetime (Indian date format handled)
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])

    # Extract month for temporal analysis
    df['month'] = df['date'].dt.to_period('M').astype(str)

    # Select only adult biometric updates (17+)
    df_17plus = df[['pincode', 'month', 'bio_age_17_']]

    # Aggregate updates at PINCODE–MONTH level
    agg_df = (
        df_17plus
        .groupby(['pincode', 'month'], as_index=False)
        .agg({'bio_age_17_': 'sum'})
        .rename(columns={'bio_age_17_': 'bio_updates_17plus'})
    )

    return agg_df
