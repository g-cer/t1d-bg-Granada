"""
Unified Hyperparameter Tuning Script for XGBoost and LightGBM
==============================================================

This script simplifies the hyperparameter tuning process by providing a unified
interface for both XGBoost and LightGBM models. It:

1. Runs Optuna optimization on train set, validates on validation set
2. Saves the best model trained on train set only
3. Generates predictions on validation set (NOT test set)
4. Saves the Optuna study for future parameter retrieval
5. Creates optimization plots and analysis

Usage:
    python unified_tuning.py --model xgb --n_trials 100
    python unified_tuning.py --model lgb --n_trials 50 --enable_gpu
    python unified_tuning.py --model both --n_trials 100 --timeout 7200
"""

import os
import argparse
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
from utils.data import (
    load_splits,
    rescale_data,
    print_results,
    calculate_metrics,
)


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["xgb", "lgb", "both"], required=True)
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/val_set")
    parser.add_argument("--models_dir", default="models/val_set")
    parser.add_argument("--study_dir", default="tuning/results")
    parser.add_argument("--splits_dir", default="data/split_sets")

    return parser.parse_args()


def calculate_patient_based_mae(df):
    """Calculate mean of MAE computed for each patient individually."""
    _, maes, _, _ = calculate_metrics(df)
    return np.mean(maes) if maes else float("inf")


def create_model_instance(model_name, seed, params):
    """Create model instance with given parameters."""
    if model_name == "xgb":
        return xgb.XGBRegressor(**params, random_state=seed, device="cuda:0")
    elif model_name == "lgb":
        return lgb.LGBMRegressor(
            **params, random_state=seed, device="gpu", verbosity=-1
        )
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def evaluate_model_on_validation(model, val_set, X_cols, y_cols):
    """Evaluate model on validation set and return patient-based MAE."""
    eval_set = val_set.copy()
    eval_set["y_pred"] = model.predict(eval_set[X_cols])
    eval_set = eval_set.rename(columns={y_cols[-1]: "target"})
    eval_set = rescale_data(eval_set, ["target", "y_pred"])

    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = eval_set[output_columns]
    return calculate_patient_based_mae(results)


def create_objective_function(model_name, train_set, val_set, X_cols, y_cols, seed):
    """Create Optuna objective function for hyperparameter optimization."""

    def objective(trial):
        if model_name == "xgb":
            trial_params = {
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 0.1, 10.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True),
            }
        elif model_name == "lgb":
            trial_params = {
                "num_leaves": trial.suggest_int("num_leaves", 10, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "min_child_weight": trial.suggest_float(
                    "min_child_weight", 1e-3, 10.0, log=True
                ),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
                "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
            }

        # Train model on train set only
        model = create_model_instance(model_name, seed, trial_params)
        model.fit(train_set[X_cols], train_set[y_cols[-1]])

        # Evaluate on validation set
        mae = evaluate_model_on_validation(model, val_set, X_cols, y_cols)
        return mae

    return objective


def run_optimization(model_name, train_set, val_set, X_cols, y_cols, args):
    """Run Optuna optimization for a specific model."""
    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER TUNING - {model_name.upper()}")
    print(f"{'='*70}")

    # Evaluate baseline performance
    print(f"Evaluating baseline {model_name.upper()} model...")
    baseline_model = create_model_instance(model_name, args.seed, {})
    baseline_model.fit(train_set[X_cols], train_set[y_cols[-1]])
    baseline_mae = evaluate_model_on_validation(baseline_model, val_set, X_cols, y_cols)
    print(f"Baseline MAE: {baseline_mae:.4f}")

    # Setup Optuna
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=args.seed),
        study_name=f"{model_name}_glucose_prediction",
    )

    # Create objective function
    objective_fn = create_objective_function(
        model_name, train_set, val_set, X_cols, y_cols, args.seed
    )

    # Run optimization
    print(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(
        objective_fn,
        n_trials=args.n_trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    # Save study
    study_path = f"{args.study_dir}/{model_name}_optuna_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Optuna study saved to: {study_path}")

    return study, baseline_mae


def train_and_evaluate_best_model(
    model_name, study, train_set, val_set, X_cols, y_cols, args
):
    """Train best model and evaluate on validation set."""
    print(
        f"\nTraining best {model_name.upper()} model and evaluating on validation set..."
    )

    best_params = study.best_params

    # Train model on train set only
    best_model = create_model_instance(model_name, args.seed, best_params)
    best_model.fit(train_set[X_cols], train_set[y_cols[-1]])

    # Save the model
    model_path = f"{args.models_dir}/{model_name}_tuned.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Best model saved to: {model_path}")

    # Generate predictions on validation set
    val_set_eval = val_set.copy()
    val_set_eval["y_pred"] = best_model.predict(val_set_eval[X_cols])
    val_set_eval = val_set_eval.rename(columns={y_cols[-1]: "target"})
    val_set_eval = rescale_data(val_set_eval, ["target", "y_pred"])

    # Select output columns
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = val_set_eval[output_columns]

    # Print and save results
    print(f"\n{model_name.upper()} Validation Set Results:")
    print_results(results)

    results_path = f"{args.output_dir}/{model_name}_tuned_output.csv"
    results.to_csv(results_path, index=False)
    print(f"Validation results saved to: {results_path}")

    return best_model, results


def main():
    """Main execution function."""
    args = parse_arguments()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 80)
    print("UNIFIED HYPERPARAMETER TUNING FOR GLUCOSE PREDICTION")
    print("=" * 80)
    print(f"Model(s): {args.model.upper()}")
    print(f"Trials: {args.n_trials}")
    print(f"Timeout: {args.timeout}s")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    # Load data
    print("\nLoading data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits(args.splits_dir)
    print(f"Train set: {train_set.shape}")
    print(f"Validation set: {val_set.shape}")
    print(f"Test set: {test_set.shape}")
    print(f"Features: {len(X_cols)}")
    print(f"Target: {y_cols}")

    # Determine models to tune
    models_to_tune = ["xgb", "lgb"] if args.model == "both" else [args.model]

    # Store results
    all_results = {}

    # Process each model
    for model_name in models_to_tune:
        try:
            # Run optimization
            study, baseline_mae = run_optimization(
                model_name, train_set, val_set, X_cols, y_cols, args
            )

            # Train best model and evaluate
            best_model, results = train_and_evaluate_best_model(
                model_name, study, train_set, val_set, X_cols, y_cols, args
            )

            # Store results
            all_results[model_name] = {
                "baseline_mae": baseline_mae,
                "best_mae": study.best_value,
                "improvement": baseline_mae - study.best_value,
                "improvement_pct": (
                    (baseline_mae - study.best_value) / baseline_mae * 100
                ),
                "best_params": study.best_params,
            }

            print(f"\n{model_name.upper()} optimization completed successfully!")

        except Exception as e:
            print(f"\nError processing {model_name.upper()}: {str(e)}")
            continue

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    for model_name, results in all_results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Baseline MAE: {results['baseline_mae']:.4f}")
        print(f"  Best MAE: {results['best_mae']:.4f}")
        print(
            f"  Improvement: {results['improvement']:.4f} ({results['improvement_pct']:.2f}%)"
        )
        print(f"  Best parameters: {results['best_params']}")
        print(f"  Files saved:")
        print(f"    - Model: {args.models_dir}/{model_name}_tuned.pkl")
        print(f"    - Study: {args.study_dir}/{model_name}_optuna_study.pkl")
        print(f"    - Results: {args.output_dir}/{model_name}_tuned_output.csv")

    print(f"\nAll optimization tasks completed!")


if __name__ == "__main__":
    main()
