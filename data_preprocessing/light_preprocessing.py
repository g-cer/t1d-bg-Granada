import pandas as pd
import numpy as np
import time
from datetime import datetime


def load_data(file_path):
    """Load and prepare the glucose measurements dataset."""
    print(f"Loading dataset '{file_path}'...")
    df_glucose = pd.read_csv(file_path)

    # Create timestamp column
    df_glucose["Timestamp"] = pd.to_datetime(
        df_glucose["Measurement_date"] + " " + df_glucose["Measurement_time"]
    )

    # Extract last 4 digits of Patient_ID
    df_glucose["Patient_ID"] = df_glucose["Patient_ID"].str[-4:]

    print(f"Dataset loaded successfully. Shape: {df_glucose.shape}")
    print(f"Unique patients: {df_glucose['Patient_ID'].nunique()}")
    return df_glucose


def resample_glucose_data(df, freq="15min", tolerance="7min"):
    """
    Resamples the glucose measurements for each patient every `freq`,
    using the closest value within `tolerance`.

    Args:
        df: DataFrame with glucose measurements
        freq: Resampling frequency (default: "15min")
        tolerance: Maximum time difference for nearest match (default: "7min")

    Returns:
        List of resampled DataFrames for each patient
    """
    print(f"Starting resampling with frequency {freq} and tolerance {tolerance}...")

    df = df.sort_values(["Patient_ID", "Timestamp"])  # essential for merge_asof
    resampled_list = []
    total_patients = df["Patient_ID"].nunique()

    for i, (patient_id, group) in enumerate(df.groupby("Patient_ID"), 1):
        if i % 10 == 0 or i == total_patients:
            print(f"Processing patient {patient_id}... ({i}/{total_patients})")

        group = group.set_index("Timestamp").sort_index()

        start = group.index.min()
        end = group.index.max()

        # Create time grid
        expected = pd.date_range(start=start, end=end, freq=freq)

        # If the last timestamp is not included in the grid, add it
        # if end not in expected:
        #     expected = expected.union([end])
        #     expected = expected.sort_values()

        expected_df = pd.DataFrame({"Timestamp": expected})
        group_reset = group[["Measurement"]].reset_index()

        # Merge closest within tolerance
        resampled = pd.merge_asof(
            expected_df,
            group_reset,
            on="Timestamp",
            direction="nearest",
            tolerance=pd.Timedelta(tolerance),
        )

        resampled["Patient_ID"] = patient_id
        resampled_list.append(resampled)

    print("Resampling completed successfully!")
    return resampled_list


def calculate_patient_statistics(resampled_data):
    """Calculate statistics for each patient from resampled data."""
    print("Calculating patient statistics...")

    amount_of_data_per_patient = pd.DataFrame(
        [
            {
                "Patient_ID": df["Patient_ID"].iloc[0],
                "Measurement_count": len(df),
                "Measurement_no_missing_values_count": df["Measurement"].count(),
            }
            for df in resampled_data
        ]
    )

    # Convert measurements to days
    minutes_in_hour = 60
    hours_in_day = 24
    samples_separation = 15

    amount_of_data_per_patient["Measurement_no_missing_values_count_days"] = (
        amount_of_data_per_patient["Measurement_no_missing_values_count"]
        * samples_separation
        / (minutes_in_hour * hours_in_day)
    )

    return amount_of_data_per_patient


def filter_patients_by_days(df_glucose_resampled, patient_stats, min_days=30):
    """Filter out patients with less than minimum days of data."""
    print(f"Filtering patients with less than {min_days} days of data...")

    patients_below_threshold = patient_stats.loc[
        patient_stats["Measurement_no_missing_values_count_days"] < min_days,
        "Patient_ID",
    ].tolist()

    print(f"Patients to be removed: {len(patients_below_threshold)}")
    print(f"Remaining patients: {len(patient_stats) - len(patients_below_threshold)}")

    patients_mask = ~df_glucose_resampled["Patient_ID"].isin(patients_below_threshold)
    filtered_data = df_glucose_resampled[patients_mask]

    return filtered_data


def filter_top_patients(df, n_patients=20):
    """Filter to keep only the top N patients with most measurements."""
    print(f"Selecting top {n_patients} patients by measurement count...")

    # Count measurements per patient
    patient_counts = df.groupby("Patient_ID").size().sort_values(ascending=False)
    top_patients = patient_counts.head(n_patients).index.tolist()

    filtered_data = df[df["Patient_ID"].isin(top_patients)]
    print(f"Selected patients: {top_patients}")
    print(
        f"Reduced dataset from {df['Patient_ID'].nunique()} to {len(top_patients)} patients"
    )

    return filtered_data


def limit_timesteps_per_patient(df, max_timesteps=1000):
    """Limit each patient to the first N timesteps."""
    print(f"Limiting each patient to first {max_timesteps} timesteps...")

    limited_data = []
    for patient_id, group in df.groupby("Patient_ID"):
        group_sorted = group.sort_values("Timestamp")
        patient_limited = group_sorted.head(max_timesteps)
        limited_data.append(patient_limited)

    result = pd.concat(limited_data, ignore_index=True)
    print(f"Reduced dataset from {len(df)} to {len(result)} measurements")

    return result


def save_preprocessed_data(data, output_path):
    """Save the preprocessed data to CSV file."""
    print(f"Saving preprocessed dataset to '{output_path}'...")
    data.to_csv(output_path, index=False)
    print("Dataset saved successfully!")


def print_final_statistics(data):
    """Print final statistics of the preprocessed dataset."""
    print("\n" + "=" * 50)
    print("PREPROCESSING RESULTS")
    print("=" * 50)
    print(f"Final dataset shape: {data.shape}")
    print(f"Number of patients: {data['Patient_ID'].nunique()}")
    print(f"Missing values: {data['Measurement'].isnull().sum()}")
    print(
        f"Missing values percentage: {data['Measurement'].isnull().sum() / len(data) * 100:.2f}%"
    )
    print("=" * 50)


def main():
    """Main preprocessing pipeline."""
    start_time = time.time()
    print(f"Starting preprocessing at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    try:
        # Load data
        df_glucose = load_data("data/T1DiabetesGranada/Glucose_measurements.csv")

        # LIGHT: Filtra per top pazienti prima del resampling
        # df_glucose = filter_top_patients(df_glucose, n_patients=10)

        # LIGHT: Limita timesteps per paziente prima del resampling
        df_glucose = limit_timesteps_per_patient(df_glucose, max_timesteps=150)

        # Resample data
        resampled_data = resample_glucose_data(df_glucose)

        # Combine resampled data
        print("Combining resampled data...")
        df_glucose_resampled = pd.concat(resampled_data, ignore_index=True)
        df_glucose_resampled = df_glucose_resampled[
            ["Patient_ID", "Timestamp", "Measurement"]
        ]

        # Calculate statistics
        patient_stats = calculate_patient_statistics(resampled_data)

        # Filter patients
        preprocessed_data = filter_patients_by_days(
            df_glucose_resampled, patient_stats, min_days=1
        )

        preprocessed_data = df_glucose_resampled

        # Save results
        save_preprocessed_data(
            preprocessed_data,
            "data/T1DiabetesGranada/Glucose_measurements_corrected.csv",
        )

        # Print final statistics
        print_final_statistics(preprocessed_data)

    except Exception as e:
        print(f"Error during preprocessing: {str(e)}")
        raise

    finally:
        end_time = time.time()
        execution_time = end_time - start_time
        print(
            f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)"
        )
        print(
            f"Preprocessing completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


if __name__ == "__main__":
    main()
