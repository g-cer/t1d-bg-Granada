import os
import numpy as np
import polars as pl
import pandas as pd
import hashlib
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    mean_absolute_percentage_error,
)

HYPO = 70.0
HYPER = 180.0
L_BOUND = 40.0
U_BOUND = 500.0


def shift_feature(cgm, horizon, col_name, shift_tolerance):
    cgm2 = cgm.clone().select(["Timestamp", "Measurement"])
    cgm2 = cgm2.rename({"Measurement": col_name})

    cgm = cgm.with_columns(
        (pl.col("Timestamp") - pl.duration(minutes=horizon)).alias("TimestampShift")
    )

    cgm = cgm.join_asof(
        cgm2,
        left_on="TimestampShift",
        right_on="Timestamp",
        tolerance=shift_tolerance,
    )
    return cgm.drop(["TimestampShift", "Timestamp_right"])


def load_transform(cgm_data, Patient_ID, horizons, shift_tolerance):
    # Filter data for specific patient
    cgm = cgm_data.filter(pl.col("Patient_ID") == Patient_ID)
    cgm = cgm.unique(subset="Timestamp").sort("Timestamp")

    # Genera colonne lag/lead con nomi non ambigui
    cgm2 = (
        cgm.clone()
        .select(["Timestamp", "Measurement"])
        .rename({"Measurement": "value"})
    )
    out = cgm.clone()

    for h in horizons:
        # h>0  => lag di h minuti (t-h)
        # h<0  => lead di |h| minuti (t+|h|)
        name = f"lag{h}" if h > 0 else (f"lead{abs(h)}" if h < 0 else "lag0")
        tmp = cgm.with_columns(
            (pl.col("Timestamp") - pl.duration(minutes=h)).alias("TimestampShift")
        )
        tmp = tmp.join_asof(
            cgm2,
            left_on="TimestampShift",
            right_on="Timestamp",
            tolerance=shift_tolerance,
        )
        out = out.join(
            tmp.select(["Timestamp", "value"]).rename({"value": name}), on="Timestamp"
        )

    out = out.drop("Measurement")
    df = out.to_pandas()

    # (usa il target futuro a +30' se presente negli horizons)
    if "lead30" in df.columns:
        df["bgClass"] = df["lead30"].apply(
            lambda x: (
                np.nan
                if pd.isna(x)
                else ("Hypo" if x < HYPO else "Hyper" if x > HYPER else "Normal")
            )
        )

    df["Patient_ID"] = Patient_ID
    df = df.dropna()

    return df.copy()


def get_cache_key(data_path, horizons, scale, shift_tolerance):
    """Generate a unique cache key based on parameters"""
    key_string = f"{data_path}_{horizons}_{scale}_{shift_tolerance}"
    return hashlib.md5(key_string.encode()).hexdigest()


def get_data(
    data_path,
    horizons,
    train_split,
    test_split,
    scale=True,
    shift_tolerance="1m",
    use_cache=True,
    cache_dir="training/data_cache",
):
    # Create cache directory if it doesn't exist
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    # Generate cache key
    cache_key = get_cache_key(data_path, horizons, scale, shift_tolerance)
    cache_file = os.path.join(cache_dir, f"processed_data_{cache_key}.parquet")

    # Try to load from cache
    if use_cache and os.path.exists(cache_file):
        print(f"Loading data from cache: {cache_file}")
        df = pd.read_parquet(cache_file)
        print(f"Loaded {len(df)} samples from cache")
    else:
        print("Processing data from scratch...")
        # Read the single CSV file containing all patients
        cgm_data = pl.read_csv(f"{data_path}/Glucose_measurements_corrected.csv")

        # Ensure Timestamp is datetime
        cgm_data = cgm_data.with_columns(pl.col("Timestamp").str.to_datetime())

        # Get all unique patient IDs from the dataset
        all_subjects = (
            cgm_data.select("Patient_ID").unique().to_pandas()["Patient_ID"].tolist()
        )
        print(f"Found {len(all_subjects)} patients in dataset")

        df = pd.DataFrame()
        for subject in sorted(all_subjects):
            if (sorted(all_subjects).index(subject) + 1) % 10 == 0 or (
                sorted(all_subjects).index(subject) == 0
            ):
                print(
                    f"Processing patient {subject} ({sorted(all_subjects).index(subject) + 1}/{len(all_subjects)})"
                )
            sub_df = load_transform(cgm_data, subject, horizons, shift_tolerance)
            df = sub_df if df.empty else pd.concat([df, sub_df])
        df = df.reset_index(drop=True)

        # Scegli le 8 lag e il target lead30
        lag_cols = [
            "lag105",
            "lag90",
            "lag75",
            "lag60",
            "lag45",
            "lag30",
            "lag15",
            "lag0",
        ]
        target_col = "lead30"

        if scale:
            df = scale_data(df, lag_cols + [target_col])

        # Save to cache
        if use_cache:
            print(f"Saving processed data to cache: {cache_file}")
            df.to_parquet(cache_file, index=False)

    # Scegli le 8 lag e il target lead30
    lag_cols = [
        "lag105",
        "lag90",
        "lag75",
        "lag60",
        "lag45",
        "lag30",
        "lag15",
        "lag0",
    ]
    target_col = "lead30"

    # Get train & test set
    train_set = df[df["Patient_ID"].isin(train_split)].reset_index(drop=True)
    test_set = df[df["Patient_ID"].isin(test_split)].reset_index(drop=True)
    print(f"Train size\t{len(train_set)}")
    print(f"Test size\t{len(test_set)}")
    print("\n")

    # Get input & output column names
    X_cols = lag_cols
    y_cols = [target_col]
    print(f"Feature columns\t{X_cols}")
    print(f"Target columns\t{y_cols}")
    print("\n")

    return train_set, test_set, X_cols, y_cols


def scale_data(df, scale_cols):
    for col in scale_cols:
        df[col] = (2 * (df[col] - L_BOUND) / (U_BOUND - L_BOUND)) - 1
    return df


def rescale_data(df, rescale_cols):
    for col in rescale_cols:
        df[col] = ((df[col] + 1) * (U_BOUND - L_BOUND) / 2) + L_BOUND
    return df


def print_results(df):
    print("Cumulative")
    samples = 0
    maes, mapes, rmses = [], [], []
    for subject in df["Patient_ID"].unique():
        x = df[df["Patient_ID"] == subject]
        samples += len(x)
        maes.append(mean_absolute_error(x["target"], x["y_pred"]))
        mapes.append(mean_absolute_percentage_error(x["target"], x["y_pred"]) * 100)
        rmses.append(root_mean_squared_error(x["target"], x["y_pred"]))
    print(f"Samples: {samples}")
    print(f"MAE: {np.mean(maes):.2f}({np.std(maes):.2f})")
    print(f"MAPE: {np.mean(mapes):.2f}({np.std(mapes):.2f})")
    print(f"RMSE: {np.mean(rmses):.2f}({np.std(rmses):.2f})")

    for condition in ["Normal", "Hyper", "Hypo"]:
        dummy = df[df["bgClass"] == condition]
        samples = 0
        maes, mapes, rmses = [], [], []
        for subject in df["Patient_ID"].unique():
            x = dummy[dummy["Patient_ID"] == subject]
            samples += len(x)
            if x.empty:
                continue
            maes.append(mean_absolute_error(x["target"], x["y_pred"]))
            mapes.append(mean_absolute_percentage_error(x["target"], x["y_pred"]) * 100)
            rmses.append(root_mean_squared_error(x["target"], x["y_pred"]))

        print("~~~~~~~~~~")
        print(f"{condition}")
        print(f"Samples: {samples}")
        print(f"MAE: {np.mean(maes):.2f}({np.std(maes):.2f})")
        print(f"MAPE: {np.mean(mapes):.2f}({np.std(mapes):.2f})")
        print(f"RMSE: {np.mean(rmses):.2f}({np.std(rmses):.2f})")
