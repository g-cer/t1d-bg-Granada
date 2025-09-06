import os
import argparse
import numpy as np
import tensorflow as tf
from utils.data import rescale_data, load_splits, print_results, calculate_metrics
from utils.tf_dnn import (
    create_model,
    prepare_data,
    train_model,
    predict_in_batches,
    print_model_summary,
)


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/val_set")
    parser.add_argument("--models_path", type=str, default="models/val_set")
    parser.add_argument(
        "--exp_name", type=str, choices=["mlp", "lstm", "gru"], required=True
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)

    return parser.parse_args()


def setup_environment(args):
    """Setup training environment"""
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.models_path, exist_ok=True)


def set_seeds(seed):
    """Set random seeds for reproducibility in TensorFlow/Keras."""
    # Set NumPy seed
    np.random.seed(seed)
    # Set TensorFlow seeds
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def calculate_patient_based_mae(df):
    """Calculate mean of MAE computed for each patient individually."""
    _, maes, _, _ = calculate_metrics(df)
    return np.mean(maes) if maes else float("inf")


def train(X_train, y_train, X_val, y_val, args):
    """Complete training and evaluation pipeline"""

    set_seeds(args.seed)

    # Create and train model
    print(f"\nCreating {args.exp_name.upper()} model...")
    model = create_model(args.exp_name)
    print_model_summary(model)

    print(f"\nTraining {args.exp_name.upper()} model...")
    model, history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        args.epochs,
        args.batch_size,
        args.lr,
        args.models_path,
        args.exp_name,
    )

    return model, history


def evaluate_model(model, model_type, eval_set, X_cols, y_cols, output_path):
    # Evaluate on validation set
    print("\nEvaluating model...")
    eval_set = eval_set.copy()  # Avoid modifying original
    eval_set["y_pred"] = predict_in_batches(model, eval_set[X_cols], model_type)
    eval_set = eval_set.rename(columns={y_cols[-1]: "target"})
    eval_set = rescale_data(eval_set, ["target", "y_pred"])

    # Select and save results
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = eval_set[output_columns]

    print_results(results)

    output_file = f"{output_path}/{model_type}_output.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")


def main():
    """Main training function"""
    args = parse_arguments()

    print(f"Starting {args.exp_name.upper()} training pipeline...")

    # Setup
    setup_environment(args)

    print("Loading pre-prepared data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    # Prepare data for TensorFlow
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        train_set, val_set, test_set, X_cols, y_cols, args.exp_name
    )

    # Train and evaluate
    model, history = train(X_train, y_train, X_val, y_val, args)

    evaluate_model(model, args.exp_name, val_set, X_cols, y_cols, args.output_path)

    print(f"\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
