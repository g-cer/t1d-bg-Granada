import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
from tuning.utils_tuning import *
from training.utils_dnn import print_model_summary, predict_in_batches
import matplotlib.pyplot as plt


def set_seeds(seed):
    """Set random seeds for reproducibility in TensorFlow/Keras."""
    # Set NumPy seed
    np.random.seed(seed)

    # Set TensorFlow seeds
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


def get_base_lstm_params():
    """Get base LSTM parameters."""
    return {
        "batch_size": 4096,
        "epochs": 100,
        "learning_rate": 0.01,
        "early_stopping_patience": 5,
        "lr_scheduler_patience": 3,
        "lr_reduction_factor": 0.1,
        "min_learning_rate": 1e-6,
        "monitor": "val_loss",
        "min_delta": 1e-5,  # min-delta is the same as the baseline
    }


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
                dropout=0.0,  # No recurrent dropout within RNN
                recurrent_dropout=0.0,  # No recurrent dropout within RNN
            )
        )

        # Add dropout only after RNN layers (except the last one) and only if dropout > 0
        # This matches PyTorch architecture: no dropout after the final RNN layer
        if return_sequences and dropout > 0:
            model.add(layers.Dropout(dropout))

    # Output layer (no dropout before Dense layer)
    model.add(layers.Dense(1))

    return model


def create_callbacks(base_params, verbose=1):
    """Create training callbacks for LSTM tuning."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=base_params["monitor"],
            patience=base_params["early_stopping_patience"],
            restore_best_weights=True,
            verbose=verbose,
            min_delta=base_params["min_delta"],
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=base_params["monitor"],
            factor=base_params["lr_reduction_factor"],
            patience=base_params["lr_scheduler_patience"],
            min_lr=base_params["min_learning_rate"],
            verbose=verbose,
        ),
    ]
    return callbacks


def reshape_data_for_lstm(X):
    """Reshape data for LSTM model (batch_size, sequence_length, features)."""
    return X.values.reshape(X.shape[0], X.shape[1], 1)


def evaluate_single_configuration(
    train_set,
    val_set,
    X_cols,
    y_cols,
    base_params,
    hidden_size,
    num_layers,
    dropout,
    seed,
    model_type="LSTM",
    verbose=1,
):
    """Evaluate a single RNN configuration and return MAE."""
    # Set seeds for reproducibility
    set_seeds(seed)

    # Suppress TensorFlow warnings during evaluation
    tf.get_logger().setLevel("ERROR")

    try:
        # Prepare data
        X_train = reshape_data_for_lstm(train_set[X_cols])
        y_train = train_set[y_cols[-1]].values
        X_val = reshape_data_for_lstm(val_set[X_cols])
        y_val = val_set[y_cols[-1]].values

        # Create model
        model = create_rnn_model(hidden_size, num_layers, dropout, model_type)

        if verbose > 0:
            print(
                f"\nConfiguration: {model_type}, hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout}"
            )
            if verbose > 1:
                print_model_summary(model)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=base_params["learning_rate"]),
            loss=keras.losses.Huber(),
            metrics=["mae"],
            steps_per_execution=256,
            run_eagerly=False,
        )

        # Create callbacks
        callbacks = create_callbacks(base_params, verbose)

        # Train model
        history = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=base_params["epochs"],
            batch_size=base_params["batch_size"],
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
        keras.backend.clear_session()

        return mae

    except Exception as e:
        print(f"Configuration failed with error: {e}")
        # Clean up on error
        keras.backend.clear_session()
        return float("inf")


def find_optimal_neurons(
    train_set,
    val_set,
    X_cols,
    y_cols,
    base_params,
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
            base_params,
            hidden_size=hidden_size,
            num_layers=2,
            dropout=0.2,
            seed=seed,
            verbose=1,
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
    base_params,
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

    # Calculate total configurations considering that 1-layer models only use dropout=0.0
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
            # For 1-layer models, only test dropout=0.0 (no dropout since there's no intermediate layer)
            if num_layers == 1:
                dropout_values = [0.0]
            else:
                dropout_values = dropout_options

            for dropout in dropout_values:
                config_count += 1
                print(
                    f"\n[{config_count}/{total_configs}] Testing: {model_type}, {num_layers} layer(s), dropout {dropout}"
                )

                mae = evaluate_single_configuration(
                    train_set,
                    val_set,
                    X_cols,
                    y_cols,
                    base_params,
                    hidden_size=best_neurons,
                    num_layers=num_layers,
                    dropout=dropout,
                    seed=seed,
                    model_type=model_type,
                    verbose=1,
                )

                # Create architecture description
                architecture = f"{model_type}-{num_layers}layer"

                all_results.append(
                    {
                        "architecture": architecture,
                        "model_type": model_type,
                        "hidden_size": best_neurons,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "mae": mae,
                    }
                )

                print(
                    f"✅ {model_type} {num_layers}-layer dropout {dropout}: MAE = {mae:.4f}"
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


def plot_tuning_results(neuron_results, grid_search_results, output_dir):
    """Create visualization plots for tuning results."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))

    # Plot 1: Neuron tuning results
    neurons = [r["hidden_size"] for r in neuron_results]
    maes = [r["mae"] for r in neuron_results]

    ax1.plot(neurons, maes, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Number of Neurons")
    ax1.set_ylabel("MAE")
    ax1.set_title("Neuron Count vs Performance")
    ax1.grid(True, alpha=0.3)

    # Highlight best neuron count
    best_idx = maes.index(min(maes))
    ax1.plot(
        neurons[best_idx],
        maes[best_idx],
        "ro",
        markersize=12,
        label=f"Best: {neurons[best_idx]} neurons",
    )
    ax1.legend()

    # Plot 2: Model type comparison
    model_performance = {}
    for result in grid_search_results:
        model_type = result["model_type"]
        if model_type not in model_performance:
            model_performance[model_type] = []
        model_performance[model_type].append(result["mae"])

    model_names = list(model_performance.keys())
    model_means = [np.mean(model_performance[model]) for model in model_names]
    model_stds = [np.std(model_performance[model]) for model in model_names]

    bars = ax2.bar(model_names, model_means, yerr=model_stds, capsize=5, alpha=0.7)
    ax2.set_xlabel("Model Type")
    ax2.set_ylabel("MAE")
    ax2.set_title("Model Type Performance Comparison")
    ax2.grid(True, alpha=0.3, axis="y")

    # Color bars
    colors = ["steelblue", "orange"]
    for i, bar in enumerate(bars):
        bar.set_color(colors[i % len(colors)])

    # Plot 3: Dropout vs Performance
    dropout_performance = {}
    for result in grid_search_results:
        dropout = result["dropout"]
        if dropout not in dropout_performance:
            dropout_performance[dropout] = []
        dropout_performance[dropout].append(result["mae"])

    dropout_values = sorted(dropout_performance.keys())
    dropout_means = [np.mean(dropout_performance[d]) for d in dropout_values]
    dropout_stds = [np.std(dropout_performance[d]) for d in dropout_values]

    ax3.errorbar(
        dropout_values,
        dropout_means,
        yerr=dropout_stds,
        marker="o",
        linewidth=2,
        markersize=8,
        capsize=5,
    )
    ax3.set_xlabel("Dropout Rate")
    ax3.set_ylabel("MAE")
    ax3.set_title("Dropout Rate vs Performance")
    ax3.grid(True, alpha=0.3)


def tune_lstm(train_set, val_set, X_cols, y_cols, args):
    """Systematic RNN hyperparameter tuning with grid search."""
    print("=" * 60)
    print("SYSTEMATIC RNN HYPERPARAMETER TUNING")
    print("=" * 60)

    # Set initial seed
    set_seeds(args.seed)

    # Set TensorFlow to only show errors during tuning
    tf.get_logger().setLevel("ERROR")

    # Get base parameters
    base_params = get_base_lstm_params()

    # Phase 1: Find optimal number of neurons using baseline LSTM architecture
    best_neurons, neuron_results = find_optimal_neurons(
        train_set, val_set, X_cols, y_cols, base_params, args.seed
    )

    # Phase 2: Grid search over architectures and model types with optimal neurons
    best_config, grid_search_results = grid_search_architectures(
        train_set, val_set, X_cols, y_cols, base_params, best_neurons, args.seed
    )

    # Save results
    save_tuning_results(neuron_results, grid_search_results, args.output_dir)

    # Create visualization plots
    plot_tuning_results(neuron_results, grid_search_results, args.output_dir)

    return best_config, baseline_mae, neuron_results, grid_search_results


if __name__ == "__main__":
    # Parse arguments using the same structure as other tuning scripts
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
    best_config, baseline_mae, neuron_results, grid_search_results = tune_lstm(
        train_set, val_set, X_cols, y_cols, args
    )

    print(f"\n{'='*60}")
    print(f"RNN TUNING SUMMARY")
    print(f"{'='*60}")
    # print(f"Baseline MAE: {baseline_mae:.4f}")
    print(f"Best configuration MAE: {best_config['mae']:.4f}")
    print(f"Best configuration:")
    print(f"  Model type: {best_config['model_type']}")
    print(f"  Architecture: {best_config['architecture']}")
    print(f"  Hidden size: {best_config['hidden_size']}")
    print(f"  Layers: {best_config['num_layers']}")
    print(f"  Dropout: {best_config['dropout']}")
    # print(
    #     f"Improvement over baseline: {((baseline_mae - best_config['mae']) / baseline_mae * 100):+.2f}%"
    # )
