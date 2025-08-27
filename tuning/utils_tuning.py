import os
import argparse
import pickle
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
import matplotlib.pyplot as plt
import seaborn as sns
from training.split_data import (
    calculate_metrics,
    rescale_data,
    load_splits,
    print_results,
)


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
        description="Tune hyperparameters for glucose prediction using patient-based MAE"
    )
    parser.add_argument(
        "--model",
        choices=["xgb", "lgb"],
        required=True,
        help="Model to tune (xgb or lgb)",
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


def run_optuna_optimization(
    train_set,
    val_set,
    X_cols,
    y_cols,
    base_params,
    objective_fn,
    n_trials,
    timeout,
    output_dir,
    seed,
    model_name,
):
    """Run Optuna hyperparameter optimization."""
    print(f"Starting Optuna hyperparameter optimization for {model_name.upper()}...")

    # Set Optuna verbosity for real-time updates
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Create study
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        study_name=f"{model_name}_glucose_prediction",
    )

    # Run optimization
    study.optimize(
        objective_fn, n_trials=n_trials, show_progress_bar=True, timeout=timeout
    )

    # Save study object
    with open(f"{output_dir}/{model_name}_optuna_study.pkl", "wb") as f:
        pickle.dump(study, f)

    print(f"\n{model_name.upper()} Optimization completed!")
    print(f"  Best patient-based MAE: {study.best_value:.4f}")
    print(f"  Best parameters: {study.best_params}")

    return study


def plot_optimization_history(study, model_name, output_dir):
    """Plot optimization history and parameter importance."""
    print(f"Generating optimization plots for {model_name.upper()}...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(
        f"{model_name.upper()} Hyperparameter Optimization Analysis",
        fontsize=16,
        fontweight="bold",
    )

    # 1. Optimization History
    ax1 = axes[0, 0]
    trials_df = study.trials_dataframe()
    ax1.plot(trials_df["number"], trials_df["value"], alpha=0.7, linewidth=1)
    ax1.axhline(
        y=study.best_value,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Best: {study.best_value:.4f}",
    )
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel("Patient-based MAE")
    ax1.set_title("Optimization History")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Parameter Importance
    try:
        importances = optuna.importance.get_param_importances(study)
        if importances:
            ax2 = axes[0, 1]
            params = list(importances.keys())
            values = list(importances.values())

            bars = ax2.barh(params, values)
            ax2.set_xlabel("Importance")
            ax2.set_title("Hyperparameter Importance")
            ax2.grid(True, alpha=0.3, axis="x")

            # Color bars
            for bar in bars:
                bar.set_color("steelblue")
                bar.set_alpha(0.7)
    except Exception as e:
        ax2 = axes[0, 1]
        ax2.text(
            0.5,
            0.5,
            "Parameter importance\nnot available",
            ha="center",
            va="center",
            transform=ax2.transAxes,
        )
        ax2.set_title("Hyperparameter Importance")

    # 3. Best trials distribution
    ax3 = axes[1, 0]
    best_trials = sorted(study.trials, key=lambda t: t.value)[
        : min(20, len(study.trials))
    ]
    best_values = [t.value for t in best_trials]
    ax3.hist(
        best_values,
        bins=min(10, len(best_values)),
        alpha=0.7,
        color="lightgreen",
        edgecolor="black",
    )
    ax3.axvline(
        x=study.best_value,
        color="red",
        linestyle="--",
        alpha=0.8,
        label=f"Best: {study.best_value:.4f}",
    )
    ax3.set_xlabel("Patient-based MAE")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Distribution of Best 20 Trials")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Convergence plot (rolling minimum)
    ax4 = axes[1, 1]
    values = [trial.value for trial in study.trials]
    rolling_min = []
    current_min = float("inf")
    for value in values:
        if value < current_min:
            current_min = value
        rolling_min.append(current_min)

    ax4.plot(range(len(rolling_min)), rolling_min, linewidth=2, color="darkblue")
    ax4.set_xlabel("Trial Number")
    ax4.set_ylabel("Best MAE So Far")
    ax4.set_title("Convergence Plot")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = f"{output_dir}/{model_name}_optimization_analysis.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Optimization plots saved to: {plot_path}")

    # Save parameter importance to CSV
    try:
        if importances:
            importance_df = pd.DataFrame(
                list(importances.items()), columns=["Parameter", "Importance"]
            )
            importance_df = importance_df.sort_values("Importance", ascending=False)
            importance_path = f"{output_dir}/{model_name}_parameter_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            print(f"Parameter importance saved to: {importance_path}")
    except:
        pass


def train_final_model_and_evaluate(
    model_class,
    best_params,
    train_set,
    val_set,
    test_set,
    X_cols,
    y_cols,
    output_dir,
    model_name,
):
    """Train final model on train+val and evaluate on test set."""
    print(f"\n{'='*60}")
    print(f"FINAL MODEL TRAINING AND EVALUATION - {model_name.upper()}")
    print(f"{'='*60}")

    # Combine train and validation sets
    train_val_set = pd.concat([train_set, val_set], ignore_index=True)
    print(f"Combined train+val set size: {len(train_val_set)}")
    print(f"Test set size: {len(test_set)}")

    # Train final model with best parameters
    print(f"\nTraining final {model_name.upper()} model with best parameters...")
    print(f"Best parameters: {best_params}")

    final_model = model_class(**best_params)
    final_model.fit(train_val_set[X_cols], train_val_set[y_cols[-1]])

    # Save the final model
    model_path = f"{output_dir}/{model_name}_best_model.pickle"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"Final model saved to: {model_path}")

    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_set_eval = test_set.copy()
    test_set_eval["y_pred"] = final_model.predict(test_set_eval[X_cols])

    # Prepare results
    test_set_eval = test_set_eval.rename(columns={y_cols[-1]: "target"})
    test_set_eval = rescale_data(test_set_eval, ["target", "y_pred"])

    # Select output columns and save
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = test_set_eval[output_columns]

    # Print results
    print_results(results)

    # Save results
    results_path = f"{output_dir}/{model_name}_test_results.csv"
    results.to_csv(results_path, index=False)
    print(f"Test results saved to: {results_path}")

    # Calculate final test MAE
    test_mae = calculate_patient_based_mae(results)
    print(f"\nFinal test MAE: {test_mae:.4f}")

    return final_model, results, test_mae
