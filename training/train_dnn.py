import os
import argparse
import tensorflow as tf
from split_data import rescale_data, load_splits, print_results
from utils_dnn import (
    create_model,
    prepare_data,
    train_model,
    predict_in_batches,
    print_model_summary,
)


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description="Train deep neural networks for glucose prediction"
    )
    parser.add_argument(
        "--data_path", default="data", type=str, help="Path to data directory"
    )
    parser.add_argument(
        "--output_path", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        choices=["mlp", "lstm", "gru"],
        help="Model name",
        required=True,
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=4096, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")

    return parser.parse_args()


def setup_environment(args):
    """Setup training environment"""
    os.makedirs(args.output_path, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)


def load_and_prepare_data(args):
    """Load and prepare all datasets"""
    print("Loading pre-prepared data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    # Prepare data for TensorFlow
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        train_set, val_set, test_set, X_cols, y_cols, args.exp_name
    )

    # Print data information
    shapes = [
        (X_train, y_train, "train"),
        (X_val, y_val, "val"),
        (X_test, y_test, "test"),
    ]
    print("Data shapes:")
    for X, y, name in shapes:
        print(f"X_{name}: {X.shape}, y_{name}: {y.shape}")

    return {
        "datasets": (train_set, val_set, test_set),
        "columns": (X_cols, y_cols),
        "arrays": (X_train, y_train, X_val, y_val, X_test, y_test),
    }


def train_and_evaluate(args, data_dict):
    """Complete training and evaluation pipeline"""
    X_train, y_train, X_val, y_val, X_test, y_test = data_dict["arrays"]
    train_set, val_set, test_set = data_dict["datasets"]
    X_cols, y_cols = data_dict["columns"]

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
        args.output_path,
        args.exp_name,
    )

    # Evaluate on validation set
    print("\nEvaluating model...")
    val_set = val_set.copy()  # Avoid modifying original
    val_set["y_pred"] = predict_in_batches(
        model, val_set[X_cols], args.exp_name, args.batch_size
    )
    val_set = val_set.rename(columns={y_cols[-1]: "target"})
    val_set = rescale_data(val_set, ["target", "y_pred"])

    # Select and save results
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = val_set[output_columns]

    print_results(results)

    output_file = f"{args.output_path}/{args.exp_name}_output.csv"
    results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")

    return model, history, results


def main():
    """Main training function"""
    args = parse_arguments()

    print(f"Starting {args.exp_name.upper()} training pipeline...")

    # Setup
    setup_environment(args)

    # Load and prepare data
    data_dict = load_and_prepare_data(args)

    # Train and evaluate
    model, history, results = train_and_evaluate(args, data_dict)

    print(f"\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
