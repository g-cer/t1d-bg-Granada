import os

# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import argparse
import tensorflow as tf
import numpy as np
import random
from utils_data import *
from utils_dnn import *


class CFG:
    l_bound = 40.0
    u_bound = 400.0
    train_split = list(map(int, np.load("data/patients/train_patients.npy")))
    test_split = list(map(int, np.load("data/patients/test_patients.npy")))
    horizons = [0, 15, 30, 45, 60, 75, 90, 105, -30]  # 8 lag + target lead30
    output_csv_header = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data", type=str)
parser.add_argument("--output_path", type=str, default="outputs")
parser.add_argument("--cache_dir", type=str, default="training/data_cache")
parser.add_argument("--use_cache", action="store_true", default=True)
parser.add_argument("--force_rebuild", action="store_true", help="Force rebuild cache")
parser.add_argument(
    "--exp_name",
    type=str,
    choices=["mlp", "lstm", "gru"],
    help="Model name. Choose from [mlp, lstm, gru]",
    required=True,
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)


if __name__ == "__main__":
    args = parser.parse_args()

    # Set seeds for reproducibility
    tf.random.set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # Use cache unless force rebuild is requested
    use_cache = args.use_cache and not args.force_rebuild

    train_set, test_set, X_cols, y_cols = get_data(
        args.data_path,
        horizons=CFG.horizons,
        train_split=CFG.train_split,
        test_split=CFG.test_split,
        use_cache=use_cache,
        cache_dir=args.cache_dir,
        scale=True,
    )

    # Prepare data for TensorFlow
    X_train, y_train, X_test, y_test = prepare_data(
        train_set, test_set, X_cols, y_cols, args.exp_name
    )

    print(f"Data shapes:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

    # Create model based on experiment type
    if args.exp_name == "mlp":
        model = create_mlp_model()
    elif args.exp_name == "lstm":
        model = create_lstm_model()
    elif args.exp_name == "gru":
        model = create_gru_model()

    # Print model summary
    print_model_summary(model)

    # Train model
    model, history = train_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
        args.epochs,
        args.batch_size,
        args.lr,
        args.output_path,
        args.exp_name,
    )

    print("\n")

    # Evaluate on test set
    test_set["y_pred"] = predict_in_batches(
        model, test_set[X_cols], args.exp_name, args.batch_size
    )
    test_set = test_set.rename(columns={y_cols[-1]: "target"})
    test_set = rescale_data(test_set, ["target", "y_pred"])
    test_set = test_set[CFG.output_csv_header]

    print_results(test_set)

    test_set.to_csv(f"{args.output_path}/{args.exp_name}_output.csv", index=False)
