"""
Refactored data splitting module for T1D Blood Glucose prediction project.

This module handles data loading, transformation, and splitting while using
utility functions from the utils module for better code organization and reusability.
"""

import os
import json
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from utils.data import (
    categorize_glucose,
    scale_data,
    print_dataset_info,
)

# Constants
HORIZONS = [0, 15, 30, 45, 60, 75, 90, 105, -30]  # 8 lag + target lead30
LAG_COLS = ["lag105", "lag90", "lag75", "lag60", "lag45", "lag30", "lag15", "lag0"]
TARGET_COL = "lead30"


def stratified_group_train_val_test_split(
    df,
    group_col="Patient_ID",
    y_col="bgClass",
    train_size=0.7,
    val_size=0.1,
    test_size=0.2,
    random_state=42,
):
    """Perform stratified group split maintaining group integrity.

    Args:
        df (pd.DataFrame): Input dataframe
        group_col (str): Column name for grouping (default: 'Patient_ID')
        y_col (str): Column name for stratification (default: 'bgClass')
        train_size (float): Proportion for training set (default: 0.7)
        val_size (float): Proportion for validation set (default: 0.1)
        test_size (float): Proportion for test set (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)

    Returns:
        tuple: (train_idx, val_idx, test_idx) - indices for each split
    """
    assert abs(train_size + val_size + test_size - 1.0) < 1e-9

    # Prepare data for splitting
    y = pd.Categorical(df[y_col]).codes.astype(np.int32)
    groups = df[group_col].values
    N = len(df)

    # Step 1: Split (train+val) vs test
    outer = StratifiedGroupKFold(
        n_splits=round(1 / test_size), shuffle=True, random_state=random_state
    )
    trainval_idx, test_idx = next(outer.split(np.zeros(N), y, groups))

    # Step 2: Split train vs val
    df_tv = df.iloc[trainval_idx]
    y_tv = y[trainval_idx]
    groups_tv = groups[trainval_idx]

    inner_val_share = val_size / (train_size + val_size)
    inner = StratifiedGroupKFold(
        n_splits=round(1 / inner_val_share), shuffle=True, random_state=random_state
    )
    tr_idx_in_tv, val_idx_in_tv = next(
        inner.split(np.zeros(len(df_tv)), y_tv, groups_tv)
    )

    # Map back to original indices
    train_idx = df_tv.index[tr_idx_in_tv].to_numpy()
    val_idx = df_tv.index[val_idx_in_tv].to_numpy()
    test_idx = df.index[test_idx].to_numpy()

    # Validate split
    validate_group_split(df, train_idx, val_idx, test_idx, group_col)

    return train_idx, val_idx, test_idx


def validate_group_split(df, train_idx, val_idx, test_idx, group_col):
    """Validate that groups are properly separated across splits.

    Args:
        df (pd.DataFrame): Input dataframe
        train_idx, val_idx, test_idx: Index arrays for each split
        group_col (str): Column name for grouping

    Raises:
        AssertionError: If groups overlap between splits
    """
    groups = [set(df.loc[idx, group_col]) for idx in [train_idx, val_idx, test_idx]]

    # Check all pairs are disjoint
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [("train", "validation"), ("train", "test"), ("validation", "test")]

    for (i, j), (name1, name2) in zip(pairs, names):
        assert groups[i].isdisjoint(groups[j]), f"{name1} and {name2} groups overlap"


def report_props(df, idx, y_col="bgClass"):
    """Report class proportions for a subset of data.

    Args:
        df (pd.DataFrame): Input dataframe
        idx: Index array for subset
        y_col (str): Column name for classes (default: 'bgClass')

    Returns:
        dict: Dictionary with class proportions
    """
    counts = df.loc[idx, y_col].value_counts(normalize=True).sort_index()
    return counts.to_dict()


def print_split_info(df, train_idx, val_idx, test_idx):
    """Print information about data splits.

    Args:
        df (pd.DataFrame): Full dataset
        train_idx, val_idx, test_idx: Index arrays for each split
    """
    total = len(df)
    proportions = [len(idx) / total for idx in [train_idx, val_idx, test_idx]]
    print(f"Split proportions (rows): {proportions}")

    print("Class distributions by set:")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        props = report_props(df, idx)
        print(f"  {name:<5}: {props}")


def save_splits(
    train_set, val_set, test_set, X_cols, y_cols, output_dir="data/split_sets"
):
    """Save datasets and metadata to files.

    Args:
        train_set, val_set, test_set (pd.DataFrame): Dataset splits
        X_cols (list): Feature column names
        y_cols (list): Target column names
        output_dir (str): Output directory path (default: 'data/split_sets')
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

    print(f"Datasets saved to {output_dir}/")
    for dataset, name in datasets:
        print(f"{name.capitalize()} set: {len(dataset)} samples")


def load_transform(cgm_data, patient_id, horizons, shift_tolerance):
    """Load and transform data for a specific patient with simplified logic.

    Args:
        cgm_data (pl.DataFrame): Polars DataFrame with CGM data
        patient_id: Patient identifier
        horizons (list): List of time horizons for lag/lead features
        shift_tolerance (str): Tolerance for timestamp matching

    Returns:
        pd.DataFrame: Transformed pandas DataFrame for the patient
    """
    # Filter and prepare base data
    cgm = (
        cgm_data.filter(pl.col("Patient_ID") == patient_id)
        .unique(subset="Timestamp")
        .sort("Timestamp")
    )

    if cgm.height == 0:
        return pd.DataFrame()

    # Create base value lookup
    cgm_values = cgm.select(["Timestamp", "Measurement"]).rename(
        {"Measurement": "value"}
    )
    result = cgm.clone()

    # Generate all lag/lead features in one loop
    for h in horizons:
        feature_name = f"lag{h}" if h > 0 else (f"lead{abs(h)}" if h < 0 else "lag0")

        # Create shifted timestamp and join
        shifted = (
            cgm.with_columns(
                (pl.col("Timestamp") - pl.duration(minutes=h)).alias("TimestampShift")
            )
            .join_asof(
                cgm_values,
                left_on="TimestampShift",
                right_on="Timestamp",
                tolerance=shift_tolerance,
            )
            .select(["Timestamp", "value"])
            .rename({"value": feature_name})
        )

        result = result.join(shifted, on="Timestamp")

    # Convert to pandas and add derived columns
    df = result.drop("Measurement").to_pandas()

    if "lead30" in df.columns:
        df["bgClass"] = df["lead30"].apply(categorize_glucose)

    df["Patient_ID"] = patient_id
    return df.dropna().copy()


def prepare_data(
    horizons=None, scale=True, data_path="data/T1DiabetesGranada", shift_tolerance="1m"
):
    """Simplified data preparation pipeline.

    Args:
        horizons (list, optional): Time horizons for features. Defaults to HORIZONS.
        scale (bool): Whether to scale the data. Defaults to True.
        data_path (str): Path to data directory. Defaults to "data/T1DiabetesGranada".
        shift_tolerance (str): Tolerance for timestamp matching. Defaults to "1m".

    Returns:
        tuple: (train_set, val_set, test_set, X_cols, y_cols)
    """
    if horizons is None:
        horizons = HORIZONS

    print("Processing data...")

    # Load data
    cgm_data = pl.read_csv(f"{data_path}/Glucose_measurements_corrected.csv")
    cgm_data = cgm_data.with_columns(pl.col("Timestamp").str.to_datetime())

    all_subjects = (
        cgm_data.select("Patient_ID").unique().to_pandas()["Patient_ID"].tolist()
    )
    print(f"Found {len(all_subjects)} patients in dataset")

    # Process all patients
    patient_dfs = []
    for i, subject in enumerate(sorted(all_subjects)):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"Processing patient {subject} ({i + 1}/{len(all_subjects)})")

        sub_df = load_transform(cgm_data, subject, horizons, shift_tolerance)
        if not sub_df.empty:
            patient_dfs.append(sub_df)

    df = pd.concat(patient_dfs, ignore_index=True)

    # Scale data if requested
    if scale:
        df = scale_data(df, LAG_COLS + [TARGET_COL])

    # Split data
    train_idx, val_idx, test_idx = stratified_group_train_val_test_split(df)

    # Print split information
    print_split_info(df, train_idx, val_idx, test_idx)

    # Create datasets
    datasets = [
        df.loc[idx].reset_index(drop=True) for idx in [train_idx, val_idx, test_idx]
    ]
    train_set, val_set, test_set = datasets

    # Print dataset information
    print_dataset_info(train_set, val_set, test_set, LAG_COLS, [TARGET_COL])

    return train_set, val_set, test_set, LAG_COLS, [TARGET_COL]


if __name__ == "__main__":
    train_set, val_set, test_set, X_cols, y_cols = prepare_data(scale=True)

    save_splits(train_set, val_set, test_set, X_cols, y_cols)

    print("Data preparation and saving completed!")
