import numpy as np
from tensorflow import keras
from keras import layers


def print_model_summary(model):
    """Print model architecture and parameter count"""
    print("\nModel Architecture:")
    model.summary()

    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")


def create_callbacks(
    early_stopping_patience=5,
    lr_scheduler=True,
    lr_reduction_factor=0.1,
    lr_scheduler_patience=3,
    min_learning_rate=1e-6,
    monitor="val_loss",
    save_best_model=True,
    model_save_path="best_model.keras",
):
    """Create training callbacks"""
    callbacks = []

    # Early stopping
    callbacks.append(
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        )
    )

    # Learning rate scheduler
    if lr_scheduler:
        callbacks.append(
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=lr_reduction_factor,
                patience=lr_scheduler_patience,
                min_lr=min_learning_rate,
                min_delta=0,
                verbose=1,
            )
        )

    # Model checkpoint
    if save_best_model:
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=model_save_path,
                monitor=monitor,
                save_best_only=True,
                verbose=1,
            )
        )

    return callbacks


def create_mlp_model():
    """
    Create MLP model following Chu et al. specifications:
    - Input: 8 features (2 hours of glucose history)
    - Architecture: Linear(8->256) -> Tanh + Dropout(0.2) -> Linear(256->256) -> Tanh + Dropout(0.2) -> Linear(256->1)
    - Parameters: ~69,126
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(8,)),
            layers.Dense(256, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(256, activation="tanh"),
            layers.Dropout(0.2),
            layers.Dense(1),  # Single output for 30 minutes ahead
        ]
    )

    return model


def create_lstm_model():
    """
    Create LSTM model following Chu et al. specifications:
    - Input: 8 timesteps x 1 feature (sequence of glucose values)
    - Architecture: LSTM(1->75, 2 layers, dropout=0.2) -> Linear(75->1)
    - Parameters: ~69,076
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(8, 1)),
            # First LSTM layer
            layers.LSTM(75, return_sequences=True),
            layers.Dropout(0.2),
            # Second LSTM layer
            layers.LSTM(75, return_sequences=False),
            # Fully connected layer
            layers.Dense(1),
        ]
    )

    return model


def create_gru_model():
    """
    Create GRU model following Chu et al. specifications:
    - Input: 8 timesteps x 1 feature (sequence of glucose values)
    - Architecture: GRU(1->86, 2 layers, dropout=0.2) -> Linear(86->1)
    - Parameters: ~67,941
    """
    model = keras.Sequential(
        [
            layers.Input(shape=(8, 1)),
            # First GRU layer
            layers.GRU(86, return_sequences=True),
            layers.Dropout(0.2),
            # Second GRU layer
            layers.GRU(86, return_sequences=False),
            # Fully connected layer
            layers.Dense(1),
        ]
    )

    return model


def reshape_for_rnn(x_train, x_test):
    """Reshape data for RNN models (LSTM/GRU)"""
    x_train_rnn = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
    x_test_rnn = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
    return x_train_rnn, x_test_rnn


def prepare_data(train_set, test_set, X_cols, y_cols, model_type):
    """Prepare data for training"""
    assert model_type in [
        "mlp",
        "lstm",
        "gru",
    ], "Model type must be one of [mlp, lstm, gru]"

    X_train = train_set[X_cols].values.astype(np.float32)
    y_train = train_set[y_cols].values.astype(np.float32)

    X_test = test_set[X_cols].values.astype(np.float32)
    y_test = test_set[y_cols].values.astype(np.float32)

    # Reshape targets to ensure single column
    if y_train.ndim > 1:
        y_train = y_train.reshape(-1, 1).flatten()
    if y_test.ndim > 1:
        y_test = y_test.reshape(-1, 1).flatten()

    # Reshape input for RNN models
    if model_type in ["lstm", "gru"]:
        X_train, X_test = reshape_for_rnn(X_train, X_test)

    return X_train, y_train, X_test, y_test


def train_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    epochs=100,
    batch_size=32,
    initial_learning_rate=0.01,
    output_path="outputs",
    exp_name="model",
):
    """Train the model with Chu et al. specifications"""

    # Compile model
    optimizer = keras.optimizers.Adam(learning_rate=initial_learning_rate)
    model.compile(optimizer=optimizer, loss=keras.losses.Huber(), metrics=["mae"])

    # Create callbacks
    callbacks = create_callbacks(
        early_stopping_patience=5,
        lr_scheduler=True,
        lr_reduction_factor=0.1,
        lr_scheduler_patience=3,
        min_learning_rate=1e-6,
        monitor="val_loss",
        save_best_model=True,
        model_save_path=f"{output_path}/{exp_name}.keras",
    )

    # Train model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # Load best weights
    model.load_weights(f"{output_path}/{exp_name}.keras")

    return model, history


def predict_in_batches(model, data, model_type="mlp", batch_size=256):
    """Make predictions in batches"""
    assert model_type in [
        "mlp",
        "lstm",
        "gru",
    ], "Model type must be one of [mlp, lstm, gru]"

    if model_type in ["lstm", "gru"]:
        # Reshape for RNN models
        data_reshaped = data.values.reshape(data.shape[0], data.shape[1], 1)
    else:
        data_reshaped = data.values

    # Make predictions
    predictions = model.predict(data_reshaped, batch_size=batch_size, verbose=0)

    return predictions
