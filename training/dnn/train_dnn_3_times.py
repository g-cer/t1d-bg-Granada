import os
import argparse
import numpy as np
import tensorflow as tf
from utils.data import rescale_data, load_splits, print_results
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
    parser.add_argument("--seed", type=int, default=42)
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


def calculate_mae(results):
    """Calculate patient-based MAE from results DataFrame"""
    patient_maes = []
    for patient_id in results["Patient_ID"].unique():
        patient_data = results[results["Patient_ID"] == patient_id]
        mae = np.mean(np.abs(patient_data["target"] - patient_data["y_pred"]))
        patient_maes.append(mae)
    return np.mean(patient_maes)


def train(X_train, y_train, X_val, y_val, args, seed):
    """Complete training and evaluation pipeline"""

    set_seeds(seed)

    # Create and train model
    print(f"\nCreating {args.exp_name.upper()} model with seed {seed}...")
    model = create_model(args.exp_name)
    if seed == args.seed:  # Print summary only for first run
        print_model_summary(model)

    print(f"\nTraining {args.exp_name.upper()} model with seed {seed}...")
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
        f"{args.exp_name}_seed_{seed}",
    )

    return model, history


def evaluate_model(model, model_type, eval_set, X_cols, y_cols):
    """Evaluate model and return results"""
    # Evaluate on validation set
    print(f"\nEvaluating model...")
    eval_set = eval_set.copy()  # Avoid modifying original
    eval_set["y_pred"] = predict_in_batches(model, eval_set[X_cols], model_type)
    eval_set = eval_set.rename(columns={y_cols[-1]: "target"})
    eval_set = rescale_data(eval_set, ["target", "y_pred"])

    # Select results columns
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = eval_set[output_columns]

    return results


def main():
    """Main training function"""
    args = parse_arguments()

    print(f"Starting {args.exp_name.upper()} training pipeline with multiple seeds...")

    # Setup
    setup_environment(args)

    print("Loading pre-prepared data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    # Prepare data for TensorFlow
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
        train_set, val_set, test_set, X_cols, y_cols, args.exp_name
    )

    # Define seeds for multiple runs
    seeds = [args.seed, args.seed + 1, args.seed + 2]

    best_mae = float("inf")
    best_model = None
    best_results = None
    best_seed = None

    print(f"\n{'='*60}")
    print(f"TRAINING WITH MULTIPLE SEEDS: {seeds}")
    print(f"{'='*60}")

    # Train with different seeds
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*40}")
        print(f"RUN {i}/3 - SEED {seed}")
        print(f"{'='*40}")

        # Train model
        model, history = train(X_train, y_train, X_val, y_val, args, seed)

        # Evaluate model
        results = evaluate_model(model, args.exp_name, val_set, X_cols, y_cols)

        # Calculate MAE
        current_mae = calculate_mae(results)
        print(f"\nSeed {seed} - Patient-based MAE: {current_mae:.4f}")

        # Check if this is the best model so far
        if current_mae < best_mae:
            best_mae = current_mae
            best_model = model
            best_results = results
            best_seed = seed
            print(f"🏆 New best model found with seed {seed}!")

        print(f"Current best MAE: {best_mae:.4f} (seed {best_seed})")

    # Save best model and results
    print(f"\n{'='*60}")
    print(f"SAVING BEST MODEL")
    print(f"{'='*60}")
    print(f"Best model achieved with seed {best_seed}")
    print(f"Best MAE: {best_mae:.4f}")

    # Print final results
    print_results(best_results)

    # Save best model (rename from temporary seed-specific name)
    import shutil

    temp_model_path = f"{args.models_path}/{args.exp_name}_seed_{best_seed}.weights.h5"
    final_model_path = f"{args.models_path}/{args.exp_name}.weights.h5"

    if os.path.exists(temp_model_path):
        shutil.move(temp_model_path, final_model_path)
        print(f"Best model saved to: {final_model_path}")

        # Clean up other temporary model files
        for seed in seeds:
            if seed != best_seed:
                temp_path = f"{args.models_path}/{args.exp_name}_seed_{seed}.weights.h5"
                if os.path.exists(temp_path):
                    os.remove(temp_path)

    # Save best results
    output_file = f"{args.output_path}/{args.exp_name}_output.csv"
    best_results.to_csv(output_file, index=False)
    print(f"Best results saved to: {output_file}")

    print(f"\nTraining pipeline completed successfully!")
    print(f"Best model trained with seed {best_seed} (MAE: {best_mae:.4f})")


if __name__ == "__main__":
    main()
