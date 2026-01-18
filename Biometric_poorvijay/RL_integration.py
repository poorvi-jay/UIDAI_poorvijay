"""
RL_integration.py
-----------------
Step D: Reinforcement Learning State Preparation

Converts biometric risk and anomaly signals into
RL-compatible state vectors.
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def prepare_rl_state(agg_df):
    """
    Builds RL state representation from biometric analytics.

    Parameters:
    agg_df (DataFrame): Anomaly-detected biometric dataset

    Returns:
    DataFrame: Normalized RL state table with identifiers
    """

    # Features observed by RL agent
    state_columns = [
        'biometric_risk',
        'anomaly_score',
        'bio_volatility',
        'bio_instability_score'
    ]

    state_df = agg_df[state_columns].copy()

    # Normalize state features to [0,1]
    scaler = MinMaxScaler()
    state_scaled = scaler.fit_transform(state_df)

    state_df = pd.DataFrame(
        state_scaled,
        columns=state_columns
    )

    # Retain identifiers for audit decisions
    state_df['pincode'] = agg_df['pincode'].values
    state_df['month'] = agg_df['month'].values

    return state_df
