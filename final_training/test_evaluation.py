import os
import pickle
import pandas as pd
import xgboost as xgb
from utils.data import load_splits, rescale_data, print_results
from utils.tf_dnn import create_gru_model, predict_in_batches


def load_best_xgb_params(study_path="tuning/results/xgb_optuna_study.pkl"):
    """Load best XGBoost hyperparameters from Optuna study."""
    print("=" * 80)
    print("LOADING BEST XGBOOST HYPERPARAMETERS")
    print("=" * 80)

    if not os.path.exists(study_path):
        raise FileNotFoundError(f"XGBoost Optuna study not found at {study_path}")

    with open(study_path, "rb") as f:
        study = pickle.load(f)

    best_params = study.best_params.copy()
    best_mae = study.best_value

    print(f"Best validation MAE from tuning: {best_mae:.4f}")
    print(f"Best hyperparameters:")
    for param, value in best_params.items():
        if isinstance(value, float):
            print(f"  {param}: {value:.6f}")
        else:
            print(f"  {param}: {value}")

    # Add fixed parameters for training
    best_params.update({"random_state": 42, "device": "cuda:0"})

    return best_params, best_mae


def train_final_xgb_model(train_set, val_set, X_cols, y_cols, best_params):
    """Train XGBoost model on combined train+val set with best hyperparameters."""
    print("\n" + "=" * 80)
    print("TRAINING FINAL XGBOOST MODEL")
    print("=" * 80)

    # Combine train and validation sets
    combined_set = pd.concat([train_set, val_set], ignore_index=True)
    print(f"Combined training set size: {len(combined_set)}")
    print(f"  - Original train set: {len(train_set)}")
    print(f"  - Original val set: {len(val_set)}")

    # Create and train model
    print(f"\nCreating XGBoost model with optimized hyperparameters...")
    model = xgb.XGBRegressor(**best_params)

    print(f"Training on combined train+val set...")
    X_combined = combined_set[X_cols]
    y_combined = combined_set[y_cols[-1]]

    model.fit(X_combined, y_combined)

    print(f"Training completed!")

    return model


def evaluate_xgb_on_test(model, test_set, X_cols, y_cols):
    """Evaluate XGBoost model on test set."""
    print("\n" + "=" * 80)
    print("EVALUATING XGBOOST ON TEST SET")
    print("=" * 80)

    # Make predictions
    test_eval = test_set.copy()
    test_eval["y_pred"] = model.predict(test_eval[X_cols])

    # Prepare results
    test_eval = test_eval.rename(columns={y_cols[-1]: "target"})
    test_eval = rescale_data(test_eval, ["target", "y_pred"])

    # Select output columns
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = test_eval[output_columns]

    # Print results
    print("XGBoost Test Set Results:")
    print_results(results)

    return results


def save_xgb_model_and_results(
    model, results, models_path="models/test_set", outputs_path="outputs/test_set"
):
    """Save XGBoost model and test results."""
    print(f"\nSaving XGBoost model and results...")

    # Create directories if they don't exist
    os.makedirs(models_path, exist_ok=True)
    os.makedirs(outputs_path, exist_ok=True)

    # Save model
    model_path = f"{models_path}/xgb.pickle"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"XGBoost model saved to: {model_path}")

    # Save results
    results_path = f"{outputs_path}/xgb_output.csv"
    results.to_csv(results_path, index=False)
    print(f"XGBoost test results saved to: {results_path}")


def load_gru_model_weights(weights_path="models/val_set/gru.weights.h5"):
    """Load GRU model with pre-trained weights."""
    print("\n" + "=" * 80)
    print("LOADING GRU MODEL")
    print("=" * 80)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"GRU weights not found at {weights_path}")

    # Create GRU model with same architecture as training
    model = create_gru_model()

    # Load weights
    model.load_weights(weights_path)
    print(f"GRU model weights loaded from: {weights_path}")

    # Print model summary
    print(f"\nGRU Model Architecture:")
    model.summary()

    return model


def evaluate_gru_on_test(model, test_set, X_cols, y_cols):
    """Evaluate GRU model on test set."""
    print("\n" + "=" * 80)
    print("EVALUATING GRU ON TEST SET")
    print("=" * 80)

    # Make predictions
    test_eval = test_set.copy()
    test_eval["y_pred"] = predict_in_batches(model, test_eval[X_cols], "gru").flatten()

    # Prepare results
    test_eval = test_eval.rename(columns={y_cols[-1]: "target"})
    test_eval = rescale_data(test_eval, ["target", "y_pred"])

    # Select output columns
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = test_eval[output_columns]

    # Print results
    print("GRU Test Set Results:")
    print_results(results)

    return results


def save_gru_results(results, outputs_path="outputs/test_set"):
    """Save GRU test results."""
    print(f"\nSaving GRU results...")

    # Create directory if it doesn't exist
    os.makedirs(outputs_path, exist_ok=True)

    # Save results
    results_path = f"{outputs_path}/gru_output.csv"
    results.to_csv(results_path, index=False)
    print(f"GRU test results saved to: {results_path}")


def main():
    """Main function to run final test evaluation."""
    print("FINAL TEST SET EVALUATION")
    print("=" * 80)
    print("Evaluating XGBoost (with optimal hyperparameters) and GRU on test set")
    print("=" * 80)

    # Load data splits
    print("Loading data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    print(f"Dataset sizes:")
    print(f"  Train set: {len(train_set)} samples")
    print(f"  Val set: {len(val_set)} samples")
    print(f"  Test set: {len(test_set)} samples")
    print(f"  Features: {len(X_cols)}")
    print(f"  Target: {y_cols}")

    # ===== XGBOOST EVALUATION =====

    # Load best hyperparameters
    best_params, best_val_mae = load_best_xgb_params()

    # Train final XGBoost model
    xgb_model = train_final_xgb_model(train_set, val_set, X_cols, y_cols, best_params)

    # Evaluate on test set
    xgb_results = evaluate_xgb_on_test(xgb_model, test_set, X_cols, y_cols)

    # Save model and results
    save_xgb_model_and_results(xgb_model, xgb_results)

    # ===== GRU EVALUATION =====

    # Load GRU model
    gru_model = load_gru_model_weights()

    # Evaluate on test set
    gru_results = evaluate_gru_on_test(gru_model, test_set, X_cols, y_cols)

    # Save results
    save_gru_results(gru_results)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Models saved in: models/test_set/")
    print(f"Results saved in: outputs/test_set/")


if __name__ == "__main__":
    main()
