"""Build the shared CPU LSTM; TensorFlow is loaded only when building a model."""

import math
import os


def build_model(input_shape, config, seed):
    """Return a compiled LSTM → Dense model for (lookback, feature_count).

    ``config`` is the complete dictionary returned by ``load_config``. Each
    call clears Keras state and resets Python, NumPy and TensorFlow randomness
    before creating weights. Reproducibility assumes the same CPU environment
    and dependency versions.
    """
    if (not isinstance(input_shape, (tuple, list)) or len(input_shape) != 2
            or any(type(size) is not int or size <= 0 for size in input_shape)):
        raise ValueError("input_shape must be (lookback, n_features), two positive integers")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")

    try:
        settings, training = config["model"], config["training"]
        units = settings["lstm_units"]
        dense_units = settings["dense_units"]
        bidirectional = settings["bidirectional"]
        optimizer = training["optimizer"]
        learning_rate = training["learning_rate"]
        loss, device = training["loss"], training["device"]
    except (KeyError, TypeError) as error:
        raise ValueError("config must contain validated model and training settings") from error
    if type(units) is not int or units <= 0:
        raise ValueError("model.lstm_units must be a positive integer")
    if type(dense_units) is not int or dense_units != 1 or bidirectional is not False:
        raise ValueError("model requires dense_units=1 and bidirectional=False")
    try:
        valid_rate = (type(learning_rate) in (int, float)
                      and math.isfinite(learning_rate) and learning_rate > 0)
    except OverflowError:
        valid_rate = False
    if not valid_rate:
        raise ValueError("training.learning_rate must be a positive finite number")
    if optimizer != "adam" or loss != "mse" or device != "cpu":
        raise ValueError("training requires optimizer='adam', loss='mse' and device='cpu'")

    # These must be set before TensorFlow's first import/runtime initialization.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    import tensorflow as tf

    try:
        tf.config.set_visible_devices([], "GPU")
    except RuntimeError as error:
        raise ValueError("CPU execution requires hiding GPUs before TensorFlow initialization") from error
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()

    with tf.device("/CPU:0"):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=tuple(input_shape)),
            tf.keras.layers.LSTM(units),
            tf.keras.layers.Dense(dense_units),
        ], name="stock_lstm")
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
        )
    return model
