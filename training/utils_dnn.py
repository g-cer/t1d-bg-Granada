import os
import numpy as np
from tensorflow import keras
from keras import layers

SUPPORTED_MODELS = ["mlp", "lstm", "gru"]
RNN_MODELS = ["lstm", "gru"]


def print_model_summary(model):
    """Print model architecture and parameter count"""
    print("\nModel Architecture:")
    model.summary()
    print(f"\nTotal parameters: {model.count_params():,}")


def create_model(model_type):
    """Create model based on type with unified interface"""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Model type must be one of {SUPPORTED_MODELS}")

    if model_type == "mlp":
        return create_mlp_model()
    elif model_type == "lstm":
        return create_lstm_model()
    elif model_type == "gru":
        return create_gru_model()


def create_mlp_model():
    """Create MLP model"""
    return keras.Sequential(
        [
            layers.Input(shape=(8,)),
            layers.Dense(256, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(1),
        ],
        name="MLP_Model",
    )


def create_lstm_model():
    """Create LSTM model"""
    return keras.Sequential(
        [
            layers.Input(shape=(8, 1)),
            layers.LSTM(75, return_sequences=True),
            layers.Dropout(0.2),
            layers.LSTM(75, return_sequences=False),
            layers.Dense(1),
        ],
        name="LSTM_Model",
    )


def create_gru_model():
    """Create GRU model"""
    return keras.Sequential(
        [
            layers.Input(shape=(8, 1)),
            layers.GRU(86, return_sequences=True),
            layers.Dropout(0.2),
            layers.GRU(86, return_sequences=False),
            layers.Dense(1),
        ],
        name="GRU_Model",
    )


def prepare_data(train_set, val_set, test_set, X_cols, y_cols, model_type):
    """Prepare data for training with simplified logic"""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Model type must be one of {SUPPORTED_MODELS}")

    # Extract and convert data in one step
    datasets = [train_set, val_set, test_set]
    X_arrays = [df[X_cols].values.astype(np.float32) for df in datasets]
    y_arrays = [df[y_cols].values.astype(np.float32) for df in datasets]

    # Ensure targets are 1D
    y_arrays = [_ensure_1d(y) for y in y_arrays]

    # Reshape for RNN models if needed
    if model_type in RNN_MODELS:
        X_arrays = [_reshape_for_rnn(X) for X in X_arrays]

    return tuple(X_arrays + y_arrays)


def _ensure_1d(array):
    """Ensure array is 1D"""
    return array.flatten() if array.ndim > 1 else array


def _reshape_for_rnn(X):
    """Reshape data for RNN models"""
    return X.reshape(X.shape[0], X.shape[1], 1)


def create_callbacks(model_save_path, **kwargs):
    """Create training callbacks with defaults"""
    defaults = {
        "early_stopping_patience": 5,
        "lr_scheduler": True,
        "lr_reduction_factor": 0.1,
        "lr_scheduler_patience": 3,
        "min_learning_rate": 1e-6,
        "monitor": "val_loss",
        "save_best_model": True,
    }
    config = {**defaults, **kwargs}

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor=config["monitor"],
            patience=config["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-5,
        )
    ]

    if config["lr_scheduler"]:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=config["monitor"],
                factor=config["lr_reduction_factor"],
                patience=config["lr_scheduler_patience"],
                min_lr=config["min_learning_rate"],
                verbose=1,
            )
        )

    if config["save_best_model"]:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor=config["monitor"],
                save_best_only=True,
                save_weights_only=True,
                verbose=1,
            )
        )

    return callbacks


def train_model(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=32,
    initial_learning_rate=0.01,
    output_path="outputs",
    exp_name="model",
):
    """Train model with simplified configuration"""

    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=initial_learning_rate),
        loss=keras.losses.Huber(),
        metrics=["mae"],
        steps_per_execution=256,
        run_eagerly=False,
    )

    # Create callbacks
    model_save_path = f"{output_path}/{exp_name}.weights.h5"
    callbacks = create_callbacks(model_save_path)

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Load best weights
    model.load_weights(model_save_path)

    return model, history


def predict_in_batches(model, data, model_type, batch_size=256):
    """Make predictions with automatic reshaping"""
    if model_type not in SUPPORTED_MODELS:
        raise ValueError(f"Model type must be one of {SUPPORTED_MODELS}")

    # Prepare data based on model type
    if model_type in RNN_MODELS:
        data_reshaped = data.values.reshape(data.shape[0], data.shape[1], 1)
    else:
        data_reshaped = data.values

    return model.predict(data_reshaped, batch_size=batch_size, verbose=0)
