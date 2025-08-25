import os
import pickle
import optuna
import optuna.visualization as vis
import numpy as np

model = "xgb"


def load_study(results_dir="tuning/results"):
    """Load only the Optuna study."""
    with open(f"{results_dir}/{model}_optuna_study.pkl", "rb") as f:
        study = pickle.load(f)
    return study


def print_essential_summary(study):
    """Print only the most important optimization information."""
    print("=" * 50)
    print("OPTIMIZATION SUMMARY")
    print("=" * 50)

    print(f"Best MAE: {study.best_value:.6f}")
    print(f"Total trials: {len(study.trials)}")
    completed_trials = len(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    )
    print(f"Completed trials: {completed_trials}")

    print(f"\nBest Parameters:")
    for param, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")


def create_essential_plots(study, save_path="tuning/plots"):
    """Create only the 3 most essential plots."""
    print(f"\nCreating essential plots in {save_path}/...")
    os.makedirs(save_path, exist_ok=True)

    # Set seed for reproducible results
    np.random.seed(42)

    # 1. OPTIMIZATION HISTORY - Shows convergence over trials
    fig = vis.plot_optimization_history(study)
    fig.write_html(f"{save_path}/{model}_optimization_history.html")
    print(f" Optimization history saved")

    # 2. PARAMETER IMPORTANCES - Shows which parameters matter most
    try:
        fig = vis.plot_param_importances(study)
        fig.write_html(f"{save_path}/{model}_param_importances.html")
        print(f" Parameter importances saved")
    except Exception as e:
        print(f" Could not create parameter importances: {e}")

    # 3. PARALLEL COORDINATE - Shows best parameter combinations
    try:
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(f"{save_path}/{model}_parallel_coordinate.html")
        print(f" Parallel coordinate plot saved")
    except Exception as e:
        print(f" Could not create parallel coordinate plot: {e}")


def main():
    """Simple main analysis."""
    # Load only the study
    study = load_study()

    # Print essential info
    print_essential_summary(study)

    # Create essential plots
    create_essential_plots(study)

    print(f"\n Analysis complete! Check tuning/plots/ for visualizations.")


if __name__ == "__main__":
    main()
