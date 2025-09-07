import os
import json
import polars as pl
import pandas as pd
from utils.data import categorize_glucose, scale_data, print_dataset_info
from data_preprocessing.split_data import (
    stratified_group_train_val_test_split,
    validate_group_split,
    print_split_info,
)

# Constants
HORIZONS = [0, 15, 30, 45, 60, 75, 90, 105, -30]  # 8 lag + target lead30
LAG_COLS = ["lag105", "lag90", "lag75", "lag60", "lag45", "lag30", "lag15", "lag0"]
TARGET_COL = "lead30"
SHIFT_TOLERANCE = "1m"


def load_glucose_with_biochemical_features(file_path):
    """Load glucose measurements enriched with biochemical features."""
    print(f"Loading enriched glucose measurements from '{file_path}'...")

    df = pl.read_csv(file_path)
    df = df.with_columns(pl.col("Timestamp").str.to_datetime())

    print(f"✓ Dataset loaded successfully. Shape: {df.shape}")
    print(f"✓ Columns: {df.columns}")

    return df


def create_shifted_feature(cgm_data, horizon, tolerance=SHIFT_TOLERANCE):
    """Create a single shifted feature using polars join_asof.

    Args:
        cgm_data (pl.DataFrame): Base glucose data
        horizon (int): Time shift in minutes (positive=lag, negative=lead)
        tolerance (str): Tolerance for timestamp matching

    Returns:
        pl.DataFrame: DataFrame with the shifted feature added
    """
    feature_name = (
        f"lag{horizon}"
        if horizon > 0
        else (f"lead{abs(horizon)}" if horizon < 0 else "lag0")
    )

    # Create lookup table for values
    value_lookup = cgm_data.select(["Timestamp", "Measurement"]).rename(
        {"Measurement": feature_name}
    )

    # Create shifted timestamps and join
    result = (
        cgm_data.with_columns(
            (pl.col("Timestamp") - pl.duration(minutes=horizon)).alias("TimestampShift")
        )
        .join_asof(
            value_lookup,
            left_on="TimestampShift",
            right_on="Timestamp",
            tolerance=tolerance,
        )
        .drop("TimestampShift")
    )

    return result


def create_windowed_features(cgm_data, patient_id, horizons=HORIZONS):
    """Create windowed features for a single patient.

    Args:
        cgm_data (pl.DataFrame): Full dataset
        patient_id: Patient identifier
        horizons (list): Time horizons for lag/lead features

    Returns:
        pd.DataFrame: Processed DataFrame with windowed features
    """
    # Filter data for specific patient
    patient_data = (
        cgm_data.filter(pl.col("Patient_ID") == patient_id)
        .unique(subset="Timestamp")
        .sort("Timestamp")
    )

    if patient_data.height == 0:
        return pd.DataFrame()

    # Start with base data (keep all columns including biochemical features)
    result = patient_data.clone()

    # Create base glucose value lookup for shifts
    glucose_values = patient_data.select(["Timestamp", "Measurement"])

    # Generate all lag/lead glucose features
    for horizon in horizons:
        feature_name = (
            f"lag{horizon}"
            if horizon > 0
            else (f"lead{abs(horizon)}" if horizon < 0 else "lag0")
        )

        # Create shifted glucose values
        shifted_values = (
            glucose_values.with_columns(
                (pl.col("Timestamp") - pl.duration(minutes=horizon)).alias(
                    "TimestampShift"
                )
            )
            .join_asof(
                glucose_values.rename({"Measurement": feature_name}),
                left_on="TimestampShift",
                right_on="Timestamp",
                tolerance=SHIFT_TOLERANCE,
            )
            .select(["Timestamp", feature_name])
        )

        # Join with result
        result = result.join(shifted_values, on="Timestamp")

    # Convert to pandas and process
    df = result.to_pandas()

    # Add glucose classification for the target
    if TARGET_COL in df.columns:
        df["bgClass"] = df[TARGET_COL].apply(categorize_glucose)

    # Remove the original measurement column as we now have lag features
    if "Measurement" in df.columns:
        df = df.drop("Measurement", axis=1)

    return df.dropna().copy()


def save_static_splits(
    train_set, val_set, test_set, X_cols, y_cols, output_dir="data/static_split_sets"
):
    """Save datasets and metadata to files in the static splits directory.

    Args:
        train_set, val_set, test_set (pd.DataFrame): Dataset splits
        X_cols (list): Feature column names
        y_cols (list): Target column names
        output_dir (str): Output directory path
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save datasets
    datasets = [(train_set, "train"), (val_set, "val"), (test_set, "test")]
    for dataset, name in datasets:
        dataset.to_parquet(f"{output_dir}/{name}_set.parquet", index=False)

    # Save metadata
    metadata = {
        "X_cols": X_cols,
        "y_cols": y_cols,
        **{f"{name}_size": len(dataset) for dataset, name in datasets},
    }

    with open(f"{output_dir}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Datasets saved to {output_dir}/")
    for dataset, name in datasets:
        print(f"   {name.capitalize()} set: {len(dataset)} samples")


def prepare_static_windowed_data(
    data_path="data/T1DiabetesGranada/Glucose_measurements_with_static.csv",
    scale=True,
    horizons=HORIZONS,
):
    """Main function to prepare windowed data with biochemical features.

    Args:
        data_path (str): Path to enriched glucose measurements
        scale (bool): Whether to scale glucose features
        horizons (list): Time horizons for lag/lead features

    Returns:
        tuple: (train_set, val_set, test_set, X_cols, y_cols)
    """
    print("🔄 PREPARING WINDOWED DATA WITH BIOCHEMICAL FEATURES")
    print("=" * 60)

    # Load enriched glucose data
    enriched_data = load_glucose_with_biochemical_features(data_path)

    # Get all patients
    all_patients = (
        enriched_data.select("Patient_ID").unique().to_pandas()["Patient_ID"].tolist()
    )
    print(f"\n📊 Found {len(all_patients)} patients in dataset")

    # Process all patients
    print("\n🔄 Creating windowed features for all patients...")
    patient_dfs = []

    for i, patient_id in enumerate(sorted(all_patients)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"   Processing patient {patient_id} ({i + 1}/{len(all_patients)})")

        patient_df = create_windowed_features(enriched_data, patient_id, horizons)
        if not patient_df.empty:
            patient_dfs.append(patient_df)

    # Combine all patient data
    df = pd.concat(patient_dfs, ignore_index=True)
    print(f"✓ Combined dataset shape: {df.shape}")

    # Identify feature columns
    lag_cols = [col for col in df.columns if "lag" in col]
    biochemical_cols = [
        col
        for col in df.columns
        if col in ["Sex", "Age", "HbA1c", "TSH", "Creatinine", "HDL", "Triglycerides"]
    ]
    static_cols = [col for col in df.columns if col in biochemical_cols]
    X_cols = lag_cols + static_cols
    y_cols = [TARGET_COL]

    print(f"\n📋 Feature summary:")
    print(f"   Lag features ({len(lag_cols)}): {lag_cols}")
    print(f"   Static features ({len(static_cols)}): {static_cols}")
    print(f"   Target column: {y_cols}")

    # Scale glucose features if requested
    if scale:
        print("\n⚖️ Scaling glucose features...")
        df = scale_data(df, lag_cols + y_cols)

    # Perform stratified group split
    print("\n🔀 Performing stratified group split...")
    train_idx, val_idx, test_idx = stratified_group_train_val_test_split(df)

    # Validate split
    validate_group_split(df, train_idx, val_idx, test_idx, "Patient_ID")
    print("✓ Group split validation passed")

    # Print split information
    print_split_info(df, train_idx, val_idx, test_idx)

    # Create datasets
    train_set = df.loc[train_idx].reset_index(drop=True)
    val_set = df.loc[val_idx].reset_index(drop=True)
    test_set = df.loc[test_idx].reset_index(drop=True)

    # Print dataset information
    print_dataset_info(train_set, val_set, test_set, X_cols, y_cols)

    return train_set, val_set, test_set, X_cols, y_cols


if __name__ == "__main__":
    # Prepare windowed data with biochemical features
    train_set, val_set, test_set, X_cols, y_cols = prepare_static_windowed_data(
        scale=True
    )

    # Save the splits to the new directory
    save_static_splits(train_set, val_set, test_set, X_cols, y_cols)

    print("\n🎉 Windowed data preparation with biochemical features completed!")
