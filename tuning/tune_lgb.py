import lightgbm as lgb
from utils_tuning import *


def get_base_lgb_params(seed, enable_gpu=False, gpu_id=0):
    """Get base LightGBM parameters."""
    params = {
        "objective": "regression",
        "metric": "mae",
        "random_state": seed,
        "verbosity": -1,
        "force_row_wise": True,
    }

    if enable_gpu:
        params.update(
            {
                "device": "gpu",  # Nuovo parametro
                "gpu_use_dp": False,  # Usa single precision per velocità
            }
        )
    return params


def create_lgb_objective_function(train_set, val_set, X_cols, y_cols, base_params):
    """Create Optuna objective function for LightGBM hyperparameter optimization."""

    def objective(trial):
        # Define hyperparameter search space optimized for glucose prediction
        trial_params = {
            # Tree structure parameters
            "num_leaves": trial.suggest_int("num_leaves", 10, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1e-3, 10.0, log=True
            ),
            # Sampling parameters
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            # Regularization parameters
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            # Learning parameters
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            # Additional parameters for time series
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 5, 50),
        }

        # Combine base params with trial params
        params = {**base_params, **trial_params}

        # Train model
        model = lgb.LGBMRegressor(**params)
        model.fit(train_set[X_cols], train_set[y_cols[-1]])

        # Evaluate model using patient-based MAE
        mae = evaluate_model(model, val_set, X_cols, y_cols)

        return mae

    return objective


def evaluate_lgb_baseline(train_set, val_set, X_cols, y_cols, base_params):
    """Evaluate baseline LightGBM model with default parameters."""
    print("Evaluating baseline LightGBM model...")

    model = lgb.LGBMRegressor(**base_params)
    model.fit(train_set[X_cols], train_set[y_cols[-1]])
    baseline_mae = evaluate_model(model, val_set, X_cols, y_cols)

    print(f"LightGBM baseline patient-based MAE: {baseline_mae:.4f}")
    return baseline_mae


def tune_lgb(train_set, val_set, X_cols, y_cols, args):
    """Tune LightGBM hyperparameters."""
    print("=" * 60)
    print("LightGBM Hyperparameter Tuning")
    print("=" * 60)

    # Get base parameters
    base_params = get_base_lgb_params(args.seed, args.enable_gpu)

    # Evaluate baseline
    baseline_mae = evaluate_lgb_baseline(
        train_set, val_set, X_cols, y_cols, base_params
    )

    # Create objective function
    objective_fn = create_lgb_objective_function(
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
        "lgb",
    )

    return study, baseline_mae
