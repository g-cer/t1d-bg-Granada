import os
import argparse
import torch
from torch import optim
from split_data import load_splits, rescale_data, print_results
from pt_utils_dnn import *


parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="data", type=str)
parser.add_argument("--output_path", type=str, default="outputs")
parser.add_argument(
    "--exp_name",
    type=str,
    choices=["mlp", "lstm", "gru"],
    help="Model name. Choose from [mlp, lstm, gru]",
    required=True,
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--batch_size", type=int, default=4096)
parser.add_argument("--hidden_size", type=int, default=256)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--activation", type=str, default="tanh")
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--min_lr", type=float, default=1e-6)
parser.add_argument("--lr_patience", type=int, default=3)
parser.add_argument("--patience", type=int, default=5)

if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    print("Loading pre-prepared data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits()

    train_loader, val_loader = get_dataloader(
        train_set, val_set, X_cols, y_cols, args.exp_name, args.batch_size
    )

    # Train model
    input_size = len(X_cols) if args.exp_name == "mlp" else 1

    if args.exp_name == "mlp":
        model = MLPModel(
            input_size,
            [args.hidden_size] * args.num_layers,
            args.dropout,
            args.activation,
        )
    else:
        model = RNNModel(
            args.exp_name, input_size, args.hidden_size, args.num_layers, args.dropout
        )

    get_params_count(model)

    criterion = nn.HuberLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.lr_patience, min_lr=args.min_lr
    )
    early_stopper = EarlyStopper(patience=args.patience)

    model = train_model(
        model,
        criterion,
        optimizer,
        scheduler,
        early_stopper,
        train_loader,
        val_loader,
        args.device,
        args.epochs,
        args.output_path,
        args.exp_name,
    )

    print("\n")

    # Evaluate
    val_set["y_pred"] = predict_in_batches(
        model, val_set[X_cols], args.device, args.exp_name
    )
    val_set = val_set.rename(columns={y_cols[-1]: "target"})
    val_set = rescale_data(val_set, ["target", "y_pred"])
    output_csv_header = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
    val_set = val_set[output_csv_header]

    print_results(val_set)

    val_set.to_csv(f"{args.output_path}/{args.exp_name}_output.csv", index=False)
