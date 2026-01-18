"""
Feature_engineering.py
----------------------
Step B: Biometric Instability Feature Engineering

This module:
1. Computes month-to-month volatility
2. Computes normalized instability score
3. Combines both into a biometric risk metric
"""

def generate_biometric_features(agg_df):
    """
    Generates instability and risk features from aggregated biometric data.

    Parameters:
    agg_df (DataFrame): PINCODE–MONTH biometric update counts

    Returns:
    DataFrame: Feature-enhanced biometric dataset
    """

    # Sort data to ensure correct time-series calculations
    agg_df = agg_df.sort_values(
        by=['pincode', 'month']
    ).reset_index(drop=True)

    # Month-to-month absolute change (volatility)
    agg_df['bio_volatility'] = (
        agg_df
        .groupby('pincode')['bio_updates_17plus']
        .diff()
        .abs()
        .fillna(0)
    )

    # Z-score based instability score
    agg_df['bio_instability_score'] = (
        agg_df
        .groupby('pincode')['bio_updates_17plus']
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )

    # Normalize volatility for fair combination
    agg_df['bio_volatility_norm'] = (
        agg_df
        .groupby('pincode')['bio_volatility']
        .transform(lambda x: (x - x.mean()) / (x.std() + 1e-6))
    )

    # Composite biometric risk score
    agg_df['biometric_risk'] = (
        0.5 * agg_df['bio_instability_score'] +
        0.5 * agg_df['bio_volatility_norm']
    )

    return agg_df
