"""
Data processing utilities for T1D Blood Glucose prediction project.

This module contains utility functions for data loading, saving, splitting,
scaling, and metric calculations that can be shared across different modules.
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

# Constants
HYPO = 70.0
HYPER = 180.0
L_BOUND = 40.0
U_BOUND = 400.0


def categorize_glucose(value):
    """Categorize glucose value into Hypo, Normal, or Hyper classes.

    Args:
        value: Glucose measurement value

    Returns:
        str: Category classification ('Hypo', 'Normal', 'Hyper')
    """
    if pd.isna(value):
        return np.nan
    elif value < HYPO:
        return "Hypo"
    elif value > HYPER:
        return "Hyper"
    else:
        return "Normal"


def scale_data(df, scale_cols):
    """Scale data to [-1, 1] range.

    Args:
        df (pd.DataFrame): DataFrame to scale
        scale_cols (list): List of column names to scale

    Returns:
        pd.DataFrame: DataFrame with scaled columns
    """
    df = df.copy()
    for col in scale_cols:
        df[col] = (2 * (df[col] - L_BOUND) / (U_BOUND - L_BOUND)) - 1
    return df


def rescale_data(df, rescale_cols):
    """Rescale data back to original range.

    Args:
        df (pd.DataFrame): DataFrame to rescale
        rescale_cols (list): List of column names to rescale

    Returns:
        pd.DataFrame: DataFrame with rescaled columns
    """
    df = df.copy()
    for col in rescale_cols:
        df[col] = ((df[col] + 1) * (U_BOUND - L_BOUND) / 2) + L_BOUND
    return df


def load_splits(splits_dir="data/split_sets"):
    """Load datasets and metadata from files.

    Args:
        splits_dir (str): Directory containing the split datasets (default: 'data/split_sets')

    Returns:
        tuple: (train_set, val_set, test_set, X_cols, y_cols)
    """
    datasets = []
    for name in ["train", "val", "test"]:
        df = pd.read_parquet(f"{splits_dir}/{name}_set.parquet")
        datasets.append(df)

    with open(f"{splits_dir}/metadata.json", "r") as f:
        metadata = json.load(f)

    return tuple(datasets + [metadata["X_cols"], metadata["y_cols"]])


def calculate_metrics(df):
    """Calculate metrics for all patients in a subset.

    Args:
        df (pd.DataFrame): DataFrame with 'Patient_ID', 'target', and 'y_pred' columns

    Returns:
        tuple: (samples, maes, mapes, rmses) - total samples and metric lists
    """
    samples = 0
    maes, mapes, rmses = [], [], []

    for patient_id in df["Patient_ID"].unique():
        patient_data = df[df["Patient_ID"] == patient_id]
        if patient_data.empty:
            continue

        samples += len(patient_data)
        maes.append(mean_absolute_error(patient_data["target"], patient_data["y_pred"]))
        mapes.append(
            mean_absolute_percentage_error(
                patient_data["target"], patient_data["y_pred"]
            )
            * 100
        )
        rmses.append(
            root_mean_squared_error(patient_data["target"], patient_data["y_pred"])
        )

    return samples, maes, mapes, rmses


def print_results(df):
    """Print evaluation results with metrics breakdown by condition.

    Args:
        df (pd.DataFrame): DataFrame with predictions and true values
    """

    def print_metrics(title, samples, maes, mapes, rmses):
        """Print formatted metrics for a specific condition."""
        if title != "Cumulative":
            print("~" * 10)
        print(title)
        print(f"Samples: {samples}")
        if maes:  # Only print if we have data
            print(f"MAE: {np.mean(maes):.2f}({np.std(maes):.2f})")
            print(f"MAPE: {np.mean(mapes):.2f}({np.std(mapes):.2f})")
            print(f"RMSE: {np.mean(rmses):.2f}({np.std(rmses):.2f})")

    # Overall results
    samples, maes, mapes, rmses = calculate_metrics(df)
    print_metrics("Cumulative", samples, maes, mapes, rmses)

    # Results by condition
    for condition in ["Normal", "Hyper", "Hypo"]:
        condition_df = df[df["bgClass"] == condition]
        samples, maes, mapes, rmses = calculate_metrics(condition_df)
        print_metrics(condition, samples, maes, mapes, rmses)


def print_dataset_info(train_set, val_set, test_set, lag_cols, target_cols):
    """Print dataset size and feature information.

    Args:
        train_set, val_set, test_set (pd.DataFrame): Dataset splits
        lag_cols (list): Feature column names
        target_cols (list): Target column names
    """
    sizes = [len(ds) for ds in [train_set, val_set, test_set]]
    names = ["Train", "Validation", "Test"]

    for name, size in zip(names, sizes):
        print(f"{name} size\t{size}")

    print(f"\nFeature columns\t{lag_cols}")
    print(f"Target columns\t{target_cols}\n")
