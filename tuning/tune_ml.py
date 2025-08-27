import os
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from utils_tuning import (
    parse_arguments,
    load_splits,
    plot_optimization_history,
    train_final_model_and_evaluate,
)
from tune_xgb import tune_xgb, get_base_xgb_params
from tune_lgb import tune_lgb, get_base_lgb_params


def main():
    """Main tuning pipeline for both XGBoost and LightGBM"""
    args = parse_arguments()

    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)

    print("=" * 60)
    print(f"{args.model.upper()} Hyperparameter Tuning for Glucose Prediction")
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

    # Run tuning based on selected model
    if args.model == "xgb":
        study, baseline_mae = tune_xgb(train_set, val_set, X_cols, y_cols, args)
        base_params = get_base_xgb_params(args.seed, args.enable_gpu)
        model_class = xgb.XGBRegressor
    elif args.model == "lgb":
        study, baseline_mae = tune_lgb(train_set, val_set, X_cols, y_cols, args)
        base_params = get_base_lgb_params(args.seed, args.enable_gpu)
        model_class = lgb.LGBMRegressor

    print(f"\n{'='*60}")
    print("TUNING SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Best MAE: {study.best_value:.4f}")
    print(
        f"Improvement: {baseline_mae - study.best_value:.4f} ({((baseline_mae - study.best_value) / baseline_mae * 100):.2f}%)"
    )
    print(f"Best parameters saved to: {args.output_dir}/{args.model}_optuna_study.pkl")

    # Generate optimization plots
    plot_optimization_history(study, args.model, args.output_dir)

    # Combine base parameters with best trial parameters
    best_params = {**base_params, **study.best_params}

    # Train final model and evaluate on test set
    final_model, test_results, test_mae = train_final_model_and_evaluate(
        model_class,
        best_params,
        train_set,
        val_set,
        test_set,
        X_cols,
        y_cols,
        args.output_dir,
        args.model,
    )

    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model.upper()}")
    print(f"Validation MAE (best trial): {study.best_value:.4f}")
    print(f"Test MAE (final model): {test_mae:.4f}")
    print(
        f"Improvement over baseline: {baseline_mae - study.best_value:.4f} ({((baseline_mae - study.best_value) / baseline_mae * 100):.2f}%)"
    )
    print(f"Final model saved to: {args.output_dir}/{args.model}_best_model.pickle")
    print(f"Test results saved to: {args.output_dir}/{args.model}_test_results.csv")
    print(
        f"Optimization plots saved to: {args.output_dir}/{args.model}_optimization_analysis.png"
    )


if __name__ == "__main__":
    main()
