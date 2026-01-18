"""
run_pipeline.py
---------------
Main execution file for the Biometric Fraud Analytics Pipeline

Runs:
Step A → Data Cleaning
Step B → Feature Engineering
Step C → Anomaly Detection
Step D → RL State Preparation
"""

from Data_cleaning import load_and_clean_biometric_data
from Feature_engineering import generate_biometric_features
from Anomaly_detection import detect_biometric_anomalies
from RL_integration import prepare_rl_state


def main():
    # -------------------------------
    # Step A: Data Cleaning
    # -------------------------------
    print("Running Step A: Data Cleaning...")
    agg_df = load_and_clean_biometric_data(
        zip_path="C:\\Users\\poorv\\UIDAI_poorvijay\\Biometric_poorvijay\\data\\api_data_aadhar_biometric.zip",
        extract_path="C:\\Users\\poorv\\UIDAI_poorvijay\\Biometric_poorvijay\\data\\biometric_updates"
    )

    # -------------------------------
    # Step B: Feature Engineering
    # -------------------------------
    print("Running Step B: Feature Engineering...")
    agg_df = generate_biometric_features(agg_df)

    # -------------------------------
    # Step C: Anomaly Detection
    # -------------------------------
    print("Running Step C: Anomaly Detection...")
    agg_df = detect_biometric_anomalies(agg_df)

    # Save anomaly output
    agg_df.to_csv("data/biometric_anomalies.csv", index=False)

    # -------------------------------
    # Step D: RL State Preparation
    # -------------------------------
    print("Running Step D: RL State Preparation...")
    state_df = prepare_rl_state(agg_df)

    # Save RL-ready state
    state_df.to_csv("data/rl_state_pincode_month.csv", index=False)

    print("\nPipeline execution completed successfully!")


if __name__ == "__main__":
    main()
