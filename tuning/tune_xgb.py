import os
import argparse
import pickle
import numpy as np
import xgboost as xgb
from utils_tuning import *


def get_base_xgb_params(seed, enable_gpu=False, gpu_id=0):
    """Get base XGBoost parameters."""
    params = {
        "objective": "reg:squarederror",
        "random_state": seed,
        "verbosity": 0,
    }

    if enable_gpu:
        params.update(
            {
                "tree_method": "gpu_hist",
                "gpu_id": gpu_id,
            }
        )

    return params


def create_xgb_objective_function(train_set, val_set, X_cols, y_cols, base_params):
    """Create Optuna objective function for XGBoost hyperparameter optimization."""

    def objective(trial):
        # Define hyperparameter search space optimized for glucose prediction
        trial_params = {
            # Tree structure parameters
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 0.1, 10.0, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            # Regularization parameters
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            # Additional parameter for time series
            "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
        }

        # Combine base params with trial params
        params = {**base_params, **trial_params}

        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(train_set[X_cols], train_set[y_cols[-1]])

        # Evaluate model using patient-based MAE
        mae = evaluate_model(model, val_set, X_cols, y_cols)

        return mae

    return objective


def evaluate_xgb_baseline(train_set, val_set, X_cols, y_cols, base_params):
    """Evaluate baseline XGBoost model with default parameters."""
    print("Evaluating baseline XGBoost model...")

    model = xgb.XGBRegressor(**base_params)
    model.fit(train_set[X_cols], train_set[y_cols[-1]])
    baseline_mae = evaluate_model(model, val_set, X_cols, y_cols)

    print(f"XGBoost baseline patient-based MAE: {baseline_mae:.4f}")
    return baseline_mae


def tune_xgb(train_set, val_set, X_cols, y_cols, args):
    """Tune XGBoost hyperparameters."""
    print("=" * 60)
    print("XGBoost Hyperparameter Tuning")
    print("=" * 60)

    # Get base parameters
    base_params = get_base_xgb_params(args.seed, args.enable_gpu)

    # Evaluate baseline
    baseline_mae = evaluate_xgb_baseline(
        train_set, val_set, X_cols, y_cols, base_params
    )

    # Create objective function
    objective_fn = create_xgb_objective_function(
        train_set, val_set, X_cols, y_cols, base_params
    )

    # Run optimization
    study = run_optuna_optimization(
        train_set,
        val_set,
        X_cols,
        y_cols,
        base_params,
        objective_fn,
        args.n_trials,
        args.timeout,
        args.output_dir,
        args.seed,
        "xgb",
    )

    return study, baseline_mae
