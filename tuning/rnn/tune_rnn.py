import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers

from utils.data import load_splits, rescale_data, calculate_metrics
from utils.tf_dnn import create_callbacks, predict_in_batches


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="tuning/results")
    parser.add_argument("--splits_dir", default="data/split_sets")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def calculate_patient_based_mae(df):
    """Calculate mean of MAE computed for each patient individually."""
    _, maes, _, _ = calculate_metrics(df)
    return np.mean(maes) if maes else float("inf")


def set_seeds(seed):
    """Set random seeds for reproducibility in TensorFlow/Keras."""
    np.random.seed(seed)
    tf.keras.backend.clear_session()
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def create_rnn_model(hidden_size, num_layers, dropout, model_type="LSTM"):
    """Create RNN model (LSTM or GRU) with specified hyperparameters."""
    model = keras.Sequential(name=f"{model_type}_Tuning_Model")

    # Input layer
    model.add(layers.Input(shape=(8, 1)))

    # Select RNN layer type
    if model_type.upper() == "LSTM":
        RNNLayer = layers.LSTM
    elif model_type.upper() == "GRU":
        RNNLayer = layers.GRU
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Use 'LSTM' or 'GRU'.")

    # RNN layers
    for i in range(num_layers):
        return_sequences = i < num_layers - 1  # Return sequences for all but last layer

        model.add(
            RNNLayer(
                hidden_size,
                return_sequences=return_sequences,
                dropout=0.0,
                recurrent_dropout=0.0,
            )
        )

        # Add dropout only after RNN layers (except the last one) and only if dropout > 0
        if return_sequences and dropout > 0:
            model.add(layers.Dropout(dropout))

    # Output layer (no dropout before Dense layer)
    model.add(layers.Dense(1))

    return model


def reshape_data_for_rnn(X):
    """Reshape data for RNN model (batch_size, sequence_length, features)."""
    return X.values.reshape(X.shape[0], X.shape[1], 1)


def evaluate_single_configuration(
    train_set,
    val_set,
    X_cols,
    y_cols,
    hidden_size,
    num_layers,
    dropout,
    seed,
    model_type="LSTM",
    verbose=0,
):
    """Evaluate a single RNN configuration and return MAE."""
    set_seeds(seed)
    tf.get_logger().setLevel("ERROR")

    try:
        # Prepare data
        X_train = reshape_data_for_rnn(train_set[X_cols])
        y_train = train_set[y_cols[-1]].values
        X_val = reshape_data_for_rnn(val_set[X_cols])
        y_val = val_set[y_cols[-1]].values

        # Create model
        model = create_rnn_model(hidden_size, num_layers, dropout, model_type)

        # Compile model (using same hyperparameters as train_dnn.py)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.01),
            loss=keras.losses.Huber(),
            metrics=["mae"],
            steps_per_execution=256,
        )

        # Create callbacks (reusing from tf_dnn.py)
        callbacks = create_callbacks()

        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=4096,
            callbacks=callbacks,
            verbose=verbose,
        )

        # Make predictions on validation set
        val_set_eval = val_set.copy()
        val_set_eval["y_pred"] = predict_in_batches(
            model, val_set[X_cols], model_type.lower()
        ).flatten()

        # Prepare results
        val_set_eval = val_set_eval.rename(columns={y_cols[-1]: "target"})
        val_set_eval = rescale_data(val_set_eval, ["target", "y_pred"])

        # Select required columns for metrics calculation
        output_columns = ["Timestamp", "Patient_ID", "bgClass", "target", "y_pred"]
        results = val_set_eval[output_columns]

        # Calculate patient-based MAE
        mae = calculate_patient_based_mae(results)

        # Clean up to avoid memory issues
        del model

        return mae

    except Exception as e:
        print(f"Configuration failed with error: {e}")
        keras.backend.clear_session()
        return float("inf")


def find_optimal_neurons(
    train_set,
    val_set,
    X_cols,
    y_cols,
    seed,
    neuron_options=[50, 64, 75, 86, 100, 128],
):
    """Find the optimal number of neurons using baseline architecture."""
    print("=" * 60)
    print("PHASE 1: Finding Optimal Number of Neurons")
    print("=" * 60)
    print(
        "Testing neuron configurations with baseline architecture (2 layers, 0.2 dropout)"
    )

    results = []

    for hidden_size in neuron_options:
        mae = evaluate_single_configuration(
            train_set,
            val_set,
            X_cols,
            y_cols,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            seed=seed,
            verbose=0,
        )

        results.append({"hidden_size": hidden_size, "mae": mae})
        print(f"Hidden size {hidden_size}: MAE = {mae:.4f}")

    # Find best configuration
    best_result = min(results, key=lambda x: x["mae"])
    best_neurons = best_result["hidden_size"]
    best_mae = best_result["mae"]

    print(f"\nBest neuron configuration: {best_neurons} neurons (MAE: {best_mae:.4f})")

    return best_neurons, results


def grid_search_architectures(
    train_set,
    val_set,
    X_cols,
    y_cols,
    best_neurons,
    seed,
    num_layers_options=[1, 2],
    model_types=["LSTM", "GRU"],
    dropout_options=[0.0, 0.1, 0.2, 0.4],
):
    """Perform grid search over layer count, model type, and dropout values."""
    print("\n" + "=" * 60)
    print("PHASE 2: Grid Search - Architecture Configurations")
    print("=" * 60)

    # Calculate total configurations
    total_1_layer_configs = len(model_types) * 1  # Only dropout=0.0 for 1-layer
    total_2_layer_configs = len(model_types) * len(
        dropout_options
    )  # All dropout values for 2-layer
    total_configs = total_1_layer_configs + total_2_layer_configs

    print(f"Testing configurations:")
    print(
        f"  1-layer models: {model_types} x dropout=[0.0] = {total_1_layer_configs} configs"
    )
    print(
        f"  2-layer models: {model_types} x dropout={dropout_options} = {total_2_layer_configs} configs"
    )
    print(f"  Total configurations: {total_configs}")

    all_results = []
    config_count = 0

    for num_layers in num_layers_options:
        for model_type in model_types:
            # For 1-layer models, only use dropout=0.0
            current_dropout_options = [0.0] if num_layers == 1 else dropout_options

            for dropout in current_dropout_options:
                config_count += 1

                mae = evaluate_single_configuration(
                    train_set,
                    val_set,
                    X_cols,
                    y_cols,
                    hidden_size=best_neurons,
                    num_layers=num_layers,
                    dropout=dropout,
                    seed=seed,
                    model_type=model_type,
                    verbose=0,
                )

                architecture = f"{num_layers}x{best_neurons}"

                result = {
                    "model_type": model_type,
                    "architecture": architecture,
                    "hidden_size": best_neurons,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "mae": mae,
                }

                all_results.append(result)

                print(
                    f"[{config_count}/{total_configs}] {model_type} {architecture} dropout={dropout}: MAE = {mae:.4f}"
                )

    # Find best overall configuration
    best_config = min(all_results, key=lambda x: x["mae"])

    print(f"\n🏆 Best overall configuration:")
    print(f"  Model type: {best_config['model_type']}")
    print(f"  Architecture: {best_config['architecture']}")
    print(f"  Hidden size: {best_config['hidden_size']}")
    print(f"  Layers: {best_config['num_layers']}")
    print(f"  Dropout: {best_config['dropout']}")
    print(f"  MAE: {best_config['mae']:.4f}")

    return best_config, all_results


def save_tuning_results(neuron_results, grid_search_results, output_dir):
    """Save tuning results to CSV files."""
    # Save neuron tuning results
    neuron_df = pd.DataFrame(neuron_results)
    neuron_df.to_csv(
        os.path.join(output_dir, "rnn_neuron_tuning_results.csv"), index=False
    )

    # Save grid search results
    grid_search_df = pd.DataFrame(grid_search_results)
    grid_search_df.to_csv(
        os.path.join(output_dir, "rnn_grid_search_results.csv"), index=False
    )

    print(f"\nResults saved to {output_dir}")


def tune_rnn(train_set, val_set, X_cols, y_cols, args):
    """Systematic RNN hyperparameter tuning with grid search."""
    print("=" * 60)
    print("SYSTEMATIC RNN HYPERPARAMETER TUNING")
    print("=" * 60)

    # Set initial seed
    set_seeds(args.seed)

    # Set TensorFlow to only show errors during tuning
    tf.get_logger().setLevel("ERROR")

    # Phase 1: Find optimal number of neurons using baseline LSTM architecture
    best_neurons, neuron_results = find_optimal_neurons(
        train_set, val_set, X_cols, y_cols, args.seed
    )

    # Phase 2: Grid search over architectures and model types with optimal neurons
    best_config, grid_search_results = grid_search_architectures(
        train_set, val_set, X_cols, y_cols, best_neurons, args.seed
    )

    # Save results
    save_tuning_results(neuron_results, grid_search_results, args.output_dir)

    return best_config, neuron_results, grid_search_results


if __name__ == "__main__":
    import pandas as pd

    # Parse arguments
    args = parse_arguments()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Load data splits
    print("Loading pre-prepared data splits...")
    train_set, val_set, test_set, X_cols, y_cols = load_splits(args.splits_dir)

    print(f"Train set size: {len(train_set)}")
    print(f"Validation set size: {len(val_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Features: {len(X_cols)}")
    print(f"Target: {y_cols}")

    # Run systematic tuning
    best_config, neuron_results, grid_search_results = tune_rnn(
        train_set, val_set, X_cols, y_cols, args
    )

    print(f"\n{'='*60}")
    print(f"RNN TUNING SUMMARY")
    print(f"{'='*60}")
    print(f"Best configuration MAE: {best_config['mae']:.4f}")
    print(f"Best configuration:")
    print(f"  Model type: {best_config['model_type']}")
    print(f"  Architecture: {best_config['architecture']}")
    print(f"  Hidden size: {best_config['hidden_size']}")
    print(f"  Layers: {best_config['num_layers']}")
    print(f"  Dropout: {best_config['dropout']}")
