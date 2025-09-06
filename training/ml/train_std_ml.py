import os
import argparse
import pickle
import lightgbm as lgb
import xgboost as xgb
from utils.data import rescale_data, load_splits, print_results

# from cuml.ensemble import RandomForestRegressor


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, default="outputs/val_set")
    parser.add_argument("--models_path", type=str, default="models/val_set")
    parser.add_argument(
        "--exp_name", type=str, choices=["lgb", "xgb", "rf"], required=True
    )
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def create_model(model_type, seed):
    """Create model based on type"""
    if model_type == "lgb":
        return lgb.LGBMRegressor(random_state=seed, device="gpu")
    elif model_type == "xgb":
        return xgb.XGBRegressor(random_state=seed, device="cuda:0")
    # elif model_type == "rf":
    #     return RandomForestRegressor(random_state=seed)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def save_model(model, model_path):
    """Save trained model"""
    with open(model_path, "wb") as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Model saved to: {model_path}")


def evaluate_and_save_results(model, val_set, X_cols, y_cols, output_path, exp_name):
    """Evaluate model and save results"""
    # Make predictions
    val_set = val_set.copy()
    val_set["y_pred"] = model.predict(val_set[X_cols])

    # Prepare results
    val_set = val_set.rename(columns={y_cols[-1]: "target"})
    val_set = rescale_data(val_set, ["target", "y_pred"])

    # Select output columns and save
    output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    results = val_set[output_columns]

    print_results(results)

    output_file = f"{output_path}/{exp_name}_output.csv"
    results.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")

    return results


def main():
    """Main training function"""
    args = parse_arguments()

    print(f"Starting {args.exp_name.upper()} training pipeline...")

    # Setup
    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.models_path, exist_ok=True)

    # Load data
    print("Loading pre-prepared data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    # Create and train model
    print(f"Creating and training {args.exp_name.upper()} model...")
    model = create_model(args.exp_name, args.seed)
    model.fit(train_set[X_cols], train_set[y_cols[-1]])

    # Save model
    model_path = f"{args.models_path}/{args.exp_name}.pickle"
    save_model(model, model_path)

    print()  # Add spacing

    # Evaluate and save results
    results = evaluate_and_save_results(
        model, val_set, X_cols, y_cols, args.output_path, args.exp_name
    )

    print(f"\nTraining pipeline completed successfully!")
    return model, results


if __name__ == "__main__":
    main()
