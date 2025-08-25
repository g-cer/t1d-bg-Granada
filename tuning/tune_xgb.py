import os
import argparse
import pickle
import numpy as np
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import sys

# Add the project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from training.split_data import calculate_metrics, rescale_data, load_splits


def calculate_patient_based_mae(df):
    """
    Calculate mean of MAE computed for each patient individually.
    This matches the evaluation logic used in print_results.
    """
    _, maes, _, _ = calculate_metrics(df)
    return np.mean(maes) if maes else float("inf")


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Tune XGBoost hyperparameters for glucose prediction using patient-based MAE"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for Optuna optimization (default: 100)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for tuning (default: 3600)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--output_dir",
        default="tuning/results",
        help="Output directory for results (default: tuning/results)",
    )
    parser.add_argument(
        "--splits_dir",
        default="training/splits",
        help="Directory with data splits (default: training/splits)",
    )
    parser.add_argument(
        "--enable_gpu",
        action="store_true",
        help="Enable GPU acceleration (requires CUDA-compatible GPU)",
    )

    return parser.parse_args()


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


def evaluate_model(model, dataset, X_cols, y_cols):
    """Evaluate model using patient-based MAE (same logic as print_results)."""
    # Make predictions on a copy to avoid modifying original data
    eval_set = dataset.copy()
    eval_set["y_pred"] = model.predict(eval_set[X_cols])

    # Prepare results using the same logic as train_ml.py
    eval_set = eval_set.rename(columns={y_cols[-1]: "target"})
    eval_set = rescale_data(eval_set, ["target", "y_pred"])

    # Select required columns for metrics calculation
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = eval_set[output_columns]

    return calculate_patient_based_mae(results)


def create_objective_function(train_set, val_set, X_cols, y_cols, base_params):
    """Create Optuna objective function for hyperparameter optimization."""

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
            # Regularization parameters (important for glucose prediction)
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            # "n_estimators": trial.suggest_int("n_estimators", 1, 5),
            # Additional parameters for time series
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


def run_optuna_optimization(
    train_set, val_set, X_cols, y_cols, base_params, n_trials, timeout, output_dir, seed
):
    """Run Optuna hyperparameter optimization."""
    print("Starting Optuna hyperparameter optimization...")

    # Set Optuna verbosity for real-time updates
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name="xgb_glucose_prediction",
    )

    # Create objective function
    objective = create_objective_function(
        train_set, val_set, X_cols, y_cols, base_params
    )

    # Run optimization
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, timeout=timeout
    )

    # Save study object
    with open(f"{output_dir}/xgb_optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    print(f"\nOptimization completed!")
    print(f"  Best patient-based MAE: {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")


def evaluate_baseline_model(train_set, val_set, X_cols, y_cols, base_params):
    """Evaluate baseline model with default parameters."""
    print("Evaluating baseline model with default XGBoost parameters...")

    model = xgb.XGBRegressor(**base_params)
    model.fit(train_set[X_cols], train_set[y_cols[-1]])
    baseline_mae = evaluate_model(model, val_set, X_cols, y_cols)

    print(f"Baseline patient-based MAE: {baseline_mae:.4f}")
    return baseline_mae


def main():
    """Main tuning pipeline"""
    args = parse_arguments()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 60)
    print("XGBoost Hyperparameter Tuning for Glucose Prediction")
    print("=" * 60)
    print(f"Method: Optuna Bayesian Optimization")
    print(f"Trials: {args.n_trials}")
    print(f"Timeout: {args.timeout}s")
    print(f"GPU enabled: {args.enable_gpu}")
    print(f"Output directory: {args.output_dir}")
    print(f"Evaluation metric: Patient-based MAE (same as print_results)")
    print("=" * 60)

    # Load data splits
    print("\nLoading data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits(args.splits_dir)

    print(f"  Train set: {train_set.shape}")
    print(f"  Validation set: {val_set.shape}")
    print(f"  Test set: {test_set.shape}")
    print(f"  Features: {len(X_cols)}")
    print(f"  Target: {y_cols}")

    # Get base parameters
    base_params = get_base_xgb_params(args.seed, args.enable_gpu)

    # Evaluate baseline model
    baseline_mae = evaluate_baseline_model(
        train_set, val_set, X_cols, y_cols, base_params
    )

    print(f"\nStarting hyperparameter optimization to improve upon baseline...")

    # Run optimization
    run_optuna_optimization(
        train_set,
        val_set,
        X_cols,
        y_cols,
        base_params,
        args.n_trials,
        args.timeout,
        args.output_dir,
        args.seed,
    )


if __name__ == "__main__":
    main()
