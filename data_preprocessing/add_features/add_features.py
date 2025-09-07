import pandas as pd


def load_datasets():
    """Load and prepare the required datasets."""
    print("Loading datasets...")

    # Load corrected datasets
    df_patients = pd.read_csv("data/T1DiabetesGranada/Patient_info_corrected.csv")
    df_glucose = pd.read_csv(
        "data/T1DiabetesGranada/Glucose_measurements_corrected.csv"
    )
    df_biochem = pd.read_csv(
        "data/T1DiabetesGranada/Biochemical_parameters_corrected.csv"
    )

    # Convert timestamp columns to datetime
    df_glucose["Timestamp"] = pd.to_datetime(df_glucose["Timestamp"])
    df_biochem["Timestamp"] = pd.to_datetime(df_biochem["Timestamp"])

    print(f"✓ Loaded {len(df_patients)} patients")
    print(f"✓ Loaded {len(df_glucose)} glucose measurements")
    print(f"✓ Loaded {len(df_biochem)} biochemical parameters")

    return df_patients, df_glucose, df_biochem


def merge_patient_data(patient_id, df_glucose, df_biochem):
    """Merge glucose and biochemical data for a single patient using backward fill."""
    # Filter data for specific patient
    patient_glucose = df_glucose[df_glucose["Patient_ID"] == patient_id]
    patient_biochem = df_biochem[df_biochem["Patient_ID"] == patient_id]

    # Skip patients without biochemical data
    if patient_biochem.empty:
        return None

    # Perform backward merge with 30-day tolerance
    # This matches each glucose measurement with the most recent biochemical data
    # within 30 days before the glucose measurement
    merged_data = pd.merge_asof(
        patient_glucose,
        patient_biochem,
        on="Timestamp",
        direction="backward",
        tolerance=pd.Timedelta(days=30),
    )

    # Remove rows with missing biochemical data
    merged_data = merged_data.dropna()

    return merged_data


def process_all_patients(df_glucose, df_biochem):
    """Process all patients and merge their glucose and biochemical data."""
    print("\nMerging glucose and biochemical data...")

    patients = df_glucose["Patient_ID"].unique()
    results_list = []

    for i, patient_id in enumerate(patients, 1):
        if i % 10 == 0 or i == len(patients):
            print(f"Processing patient {patient_id} ({i}/{len(patients)})")

        patient_result = merge_patient_data(patient_id, df_glucose, df_biochem)

        if patient_result is not None:
            results_list.append(patient_result)

    # Combine all patient results
    combined_data = pd.concat(results_list, ignore_index=True)

    # Clean up duplicate Patient_ID columns from merge
    combined_data.drop(columns="Patient_ID_y", inplace=True)
    combined_data.rename(columns={"Patient_ID_x": "Patient_ID"}, inplace=True)

    print(f"✓ Successfully merged data for {len(combined_data)} measurements")

    return combined_data


def add_static_features(glucose_biochem_data, df_patients):
    """Add static patient features (age, sex) to the dataset."""
    print("\nAdding static patient features...")

    # Merge with patient demographics
    full_dataset = glucose_biochem_data.merge(df_patients, on="Patient_ID", how="left")

    # Reorder columns for better readability
    ordered_cols = ["Patient_ID", "Sex", "Age", "Timestamp", "Measurement"]
    other_cols = [col for col in full_dataset.columns if col not in ordered_cols]
    full_dataset = full_dataset[ordered_cols + other_cols]

    print(f"✓ Final dataset shape: {full_dataset.shape}")

    return full_dataset


def save_final_dataset(dataset, output_path):
    """Save the final enriched dataset."""
    print(f"\nSaving final dataset to: {output_path}")
    dataset.to_csv(output_path, index=False)
    print("✓ Dataset saved successfully")


if __name__ == "__main__":
    print("🔄 ADDING FEATURES TO GLUCOSE MEASUREMENTS")
    print("=" * 50)

    # Load all required datasets
    df_patients, df_glucose, df_biochem = load_datasets()

    # Merge glucose and biochemical data for all patients
    glucose_biochem_data = process_all_patients(df_glucose, df_biochem)

    # Add static patient features
    final_dataset = add_static_features(glucose_biochem_data, df_patients)

    # Save the enriched dataset
    save_final_dataset(
        final_dataset,
        "data/T1DiabetesGranada/Glucose_measurements_with_static.csv",
    )

    print("\n✅ Feature addition completed!")
