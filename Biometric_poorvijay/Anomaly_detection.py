"""
Anomaly_detection.py
--------------------
Step C: Unsupervised Anomaly Detection

Uses Isolation Forest to detect abnormal biometric behavior
without predefined fraud labels.
"""

from sklearn.ensemble import IsolationForest


def detect_biometric_anomalies(agg_df):
    """
    Identifies anomalous biometric activity using Isolation Forest.

    Parameters:
    agg_df (DataFrame): Feature-engineered biometric dataset

    Returns:
    DataFrame: Dataset with anomaly flags and scores
    """

    # Select behavior-based features only
    features = agg_df[
        ['bio_instability_score', 'bio_volatility']
    ].copy()

    # Initialize Isolation Forest
    model = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # Top 5% treated as anomalies
        random_state=42
    )

    # Fit model and predict anomalies
    agg_df['anomaly_flag'] = model.fit_predict(features)

    # Anomaly score (lower = more anomalous)
    agg_df['anomaly_score'] = model.decision_function(features)

    return agg_df
