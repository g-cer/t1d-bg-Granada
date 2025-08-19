import argparse, os, pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

from training.utils_data import get_data
from training.utils_dnn import predict_in_batches

TREE_MODELS = {"xgb", "lgb"}  # utilizzano shap.Tree-explainer
NN_MODELS = {"mlp", "gru", "lstm"}  # utilizzano shap.KernelExplainer


def load_model(model_name, model_path):
    """Load model based on its type"""
    if model_name in TREE_MODELS:
        with open(model_path, "rb") as f:
            return pickle.load(f)
    elif model_name in NN_MODELS:
        return keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def create_explainer(model, model_name, X_bg):
    """Create appropriate SHAP explainer based on model type"""
    if model_name in TREE_MODELS:
        return shap.TreeExplainer(model, X_bg)
    elif model_name in NN_MODELS:
        # For neural networks, create a wrapper function
        def model_predict(X):
            return predict_in_batches(
                model, pd.DataFrame(X, columns=X_bg.columns), model_type=model_name
            )

        return shap.KernelExplainer(model_predict, X_bg.values)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default="data")
    ap.add_argument("--cache_dir", default="training/data_cache")
    ap.add_argument(
        "--model_name", choices=["xgb", "lgb", "mlp", "gru", "lstm"], required=True
    )
    ap.add_argument("--n_background", type=int, default=2000)
    ap.add_argument("--n_samples", type=int, default=10000)
    ap.add_argument("--outdir", default="xai_outputs")
    args = ap.parse_args()

    # Set model path
    if args.model_name in TREE_MODELS:
        args.model_path = f"outputs/{args.model_name}.pickle"
    elif args.model_name in NN_MODELS:
        args.model_path = f"outputs/{args.model_name}.keras"

    os.makedirs(args.outdir, exist_ok=True)

    # Carica i dati (stesse finestre e scaling dei training script)
    train_set, test_set, X_cols, y_cols = get_data(
        data_path=args.data_path,
        horizons=[0, 15, 30, 45, 60, 75, 90, 105, -30],
        train_split=list(map(int, np.load("data/patients/train_patients.npy"))),
        test_split=list(map(int, np.load("data/patients/test_patients.npy"))),
        use_cache=True,
        cache_dir=args.cache_dir,
        scale=True,
    )

    model = load_model(args.model_name, args.model_path)
    print(f"Loaded {args.model_name} model from {args.model_path}")

    # Campioni per SHAP
    if args.model_name in NN_MODELS:
        # Reduce samples for neural networks due to computational cost
        n_bg = min(args.n_background, 100)
        n_eval = min(args.n_samples, 500)
        print(
            f"Using reduced sample sizes for neural network: bg={n_bg}, eval={n_eval}"
        )
    else:
        n_bg = min(args.n_background, len(train_set))
        n_eval = min(args.n_samples, len(test_set))

    X_bg = train_set[X_cols].sample(n_bg, random_state=42)
    X_eval = test_set[X_cols].sample(n_eval, random_state=42)

    # Create SHAP explainer
    print(f"Creating SHAP explainer for {args.model_name}...")
    explainer = create_explainer(model, args.model_name, X_bg)

    # Calculate SHAP values
    print("Computing SHAP values...")
    if args.model_name in NN_MODELS:
        # For neural networks, this might take longer
        shap_values = explainer.shap_values(X_eval.values)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]  # For multi-output models, take first output
        if len(shap_values.shape) == 3:  # Handle extra dimension for neural networks
            shap_values = shap_values.squeeze(-1)  # Remove last dimension if it's 1
    else:
        shap_values = explainer.shap_values(X_eval)

    # Summary plot
    plt.figure()
    shap.summary_plot(shap_values, X_eval, show=False)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.outdir, f"{args.model_name}_shap_summary.png"), dpi=200
    )
    plt.close()

    # Bar plot con importanze medie assolute
    mean_abs = np.abs(shap_values).mean(axis=0)
    imp = pd.Series(mean_abs, index=X_cols).sort_values(ascending=False)
    imp.to_csv(os.path.join(args.outdir, f"{args.model_name}_shap_importances.csv"))
    ax = imp.plot(kind="bar")
    ax.figure.tight_layout()
    ax.figure.savefig(
        os.path.join(args.outdir, f"{args.model_name}_shap_bar.png"), dpi=200
    )
    plt.close()

    print("Top-5 feature per SHAP (|mean|):")
    print(imp.head(5))
    print(
        f"\nSHAP analysis completed for {args.model_name}. Results saved in {args.outdir}/"
    )


if __name__ == "__main__":
    main()
