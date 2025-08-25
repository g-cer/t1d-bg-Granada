import os
import json
import numpy as np
import polars as pl
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

HYPO = 70.0
HYPER = 180.0
L_BOUND = 40.0
U_BOUND = 400.0
HORIZONS = [0, 15, 30, 45, 60, 75, 90, 105, -30]  # 8 lag + target lead30
LAG_COLS = ["lag105", "lag90", "lag75", "lag60", "lag45", "lag30", "lag15", "lag0"]
TARGET_COL = "lead30"


def load_transform(cgm_data, patient_id, horizons, shift_tolerance):
    """Load and transform data for a specific patient with simplified logic"""
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
        df["bgClass"] = df["lead30"].apply(_categorize_glucose)

    df["Patient_ID"] = patient_id
    return df.dropna().copy()


def _categorize_glucose(value):
    """Categorize glucose value"""
    if pd.isna(value):
        return np.nan
    elif value < HYPO:
        return "Hypo"
    elif value > HYPER:
        return "Hyper"
    else:
        return "Normal"


def prepare_data(horizons=None, scale=True, data_path="data", shift_tolerance="1m"):
    """Simplified data preparation pipeline"""
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
    _print_split_info(df, train_idx, val_idx, test_idx)

    # Create datasets
    datasets = [
        df.loc[idx].reset_index(drop=True) for idx in [train_idx, val_idx, test_idx]
    ]
    train_set, val_set, test_set = datasets

    # Print dataset information
    _print_dataset_info(train_set, val_set, test_set)

    return train_set, val_set, test_set, LAG_COLS, [TARGET_COL]


def _print_split_info(df, train_idx, val_idx, test_idx):
    """Print split information"""
    total = len(df)
    proportions = [len(idx) / total for idx in [train_idx, val_idx, test_idx]]
    print(f"Split proportions (rows): {proportions}")

    print("Class distributions by set:")
    for name, idx in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        props = report_props(df, idx)
        print(f"  {name:<5}: {props}")


def _print_dataset_info(train_set, val_set, test_set):
    """Print dataset size information"""
    sizes = [len(ds) for ds in [train_set, val_set, test_set]]
    names = ["Train", "Validation", "Test"]

    for name, size in zip(names, sizes):
        print(f"{name} size\t{size}")

    print(f"\nFeature columns\t{LAG_COLS}")
    print(f"Target columns\t{[TARGET_COL]}\n")


def stratified_group_train_val_test_split(
    df,
    group_col="Patient_ID",
    y_col="bgClass",
    train_size=0.7,
    val_size=0.1,
    test_size=0.2,
    random_state=42,
):
    """Stratified group split with simplified logic"""
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
    _validate_group_split(df, train_idx, val_idx, test_idx, group_col)

    return train_idx, val_idx, test_idx


def _validate_group_split(df, train_idx, val_idx, test_idx, group_col):
    """Validate that groups are properly separated"""
    groups = [set(df.loc[idx, group_col]) for idx in [train_idx, val_idx, test_idx]]

    # Check all pairs are disjoint
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [("train", "validation"), ("train", "test"), ("validation", "test")]

    for (i, j), (name1, name2) in zip(pairs, names):
        assert groups[i].isdisjoint(groups[j]), f"{name1} and {name2} groups overlap"


def report_props(df, idx, y_col="bgClass"):
    """Report class proportions"""
    counts = df.loc[idx, y_col].value_counts(normalize=True).sort_index()
    return counts.to_dict()


def scale_data(df, scale_cols):
    """Scale data to [-1, 1] range"""
    df = df.copy()
    for col in scale_cols:
        df[col] = (2 * (df[col] - L_BOUND) / (U_BOUND - L_BOUND)) - 1
    return df


def rescale_data(df, rescale_cols):
    """Rescale data back to original range"""
    df = df.copy()
    for col in rescale_cols:
        df[col] = ((df[col] + 1) * (U_BOUND - L_BOUND) / 2) + L_BOUND
    return df


def save_splits(
    train_set, val_set, test_set, X_cols, y_cols, output_dir="training/splits"
):
    """Save datasets and metadata"""
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


def load_splits(splits_dir="training/splits"):
    """Load datasets and metadata"""
    datasets = []
    for name in ["train", "val", "test"]:
        df = pd.read_parquet(f"{splits_dir}/{name}_set.parquet")
        datasets.append(df)

    with open(f"{splits_dir}/metadata.json", "r") as f:
        metadata = json.load(f)

    return tuple(datasets + [metadata["X_cols"], metadata["y_cols"]])


def calculate_metrics(df):
    """Calculate metrics for all patients in subset"""
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
    """Print evaluation results with simplified logic"""

    def print_metrics(title, samples, maes, mapes, rmses):
        """Print formatted metrics"""
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


if __name__ == "__main__":
    train_set, val_set, test_set, X_cols, y_cols = prepare_data()

    save_splits(train_set, val_set, test_set, X_cols, y_cols)

    print("Data preparation and saving completed!")
