import copy
import os
from pathlib import Path
import random
import subprocess
import sys
import unittest

import numpy as np

from lstm_comparison.config import load_config
from lstm_comparison.model import build_model


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "research.json"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools" / "matplotlib"))


class ModelTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config(CONFIG)

    def test_import_does_not_load_tensorflow_or_keras(self):
        result = subprocess.run([
            sys.executable, "-c",
            "import sys; import lstm_comparison.model; "
            "assert not any(k.split('.')[0] in {'tensorflow', 'keras'} for k in sys.modules)",
        ], cwd=ROOT, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_invalid_shapes_and_seeds_are_rejected_before_tensorflow_import(self):
        for shape in (None, 60, (), (60,), (60, 1, 1), (0, 1), (-1, 5),
                      (60, 0), (60.0, 1), (True, 1), (60, "5")):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "input_shape"):
                build_model(shape, self.config, 42)
        for seed in (None, True, -1, 2**32, 42.0, "42"):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                build_model((60, 1), self.config, seed)

    def test_all_feature_widths_have_shared_architecture_and_can_train(self):
        original = copy.deepcopy(self.config)
        rng = np.random.default_rng(123)
        for width, parameters in ((1, 16961), (5, 17985), (10, 19265)):
            with self.subTest(width=width):
                model = build_model((60, width), self.config, 42)
                import tensorflow as tf

                self.assertEqual(model.input_shape, (None, 60, width))
                self.assertEqual(model.output_shape, (None, 1))
                self.assertEqual(len(model.layers), 2)
                self.assertIsInstance(model.layers[0], tf.keras.layers.LSTM)
                self.assertEqual(model.layers[0].units, 64)
                self.assertFalse(model.layers[0].go_backwards)
                self.assertFalse(model.layers[0].return_sequences)
                self.assertIsInstance(model.layers[1], tf.keras.layers.Dense)
                self.assertEqual(model.layers[1].units, 1)
                self.assertEqual(model.layers[1].activation.__name__, "linear")
                self.assertEqual(model.count_params(), parameters)
                self.assertIsInstance(model.optimizer, tf.keras.optimizers.Adam)
                self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.001)
                self.assertEqual(model.loss, "mse")
                self.assertEqual(tf.config.get_visible_devices("GPU"), [])
                for weight in model.weights:
                    self.assertIn("CPU", weight.handle.device.upper())

                X = rng.uniform(0, 1, (16, 60, width)).astype("float32")
                y = (0.8 * X[:, -1, :1] + 0.1).astype("float32")
                # Explicit datasets avoid Keras creating a large private pool
                # for each array-backed fit/validation call on constrained CPUs.
                options = tf.data.Options()
                options.threading.private_threadpool_size = 1
                options.threading.max_intra_op_parallelism = 1
                train = tf.data.Dataset.from_tensor_slices((X[:12], y[:12])).batch(4).with_options(options)
                validation = tf.data.Dataset.from_tensor_slices((X[12:], y[12:])).batch(4).with_options(options)
                initial = [weight.copy() for weight in model.get_weights()]
                history = model.fit(train, validation_data=validation, epochs=2, shuffle=False, verbose=0)
                for key in ("loss", "val_loss"):
                    self.assertEqual(len(history.history[key]), 2)
                    self.assertTrue(np.isfinite(history.history[key]).all())
                predictions = model(X[12:], training=False).numpy()
                self.assertEqual(predictions.shape, (4, 1))
                self.assertTrue(np.isfinite(predictions).all())
                self.assertTrue(any(not np.array_equal(before, after)
                                    for before, after in zip(initial, model.get_weights())))
        self.assertEqual(self.config, original)

    def test_seed_resets_initial_weights_and_python_numpy_tensorflow_rngs(self):
        first = build_model((60, 5), self.config, 42)
        import tensorflow as tf

        weights = [weight.copy() for weight in first.get_weights()]
        first_draw = (random.random(), np.random.random(), tf.random.uniform((3,)).numpy())
        second = build_model((60, 5), self.config, 42)
        second_draw = (random.random(), np.random.random(), tf.random.uniform((3,)).numpy())
        for left, right in zip(weights, second.get_weights()):
            np.testing.assert_array_equal(left, right)
        self.assertEqual(first_draw[:2], second_draw[:2])
        np.testing.assert_array_equal(first_draw[2], second_draw[2])
        third = build_model((60, 5), self.config, 43)
        self.assertTrue(any(not np.array_equal(left, right)
                            for left, right in zip(weights, third.get_weights())))
        # A stable automatic layer name also verifies old Keras state is cleared.
        self.assertEqual(first.layers[0].name, third.layers[0].name)

    def test_model_uses_configured_units_and_learning_rate(self):
        self.config["model"]["lstm_units"] = 8
        self.config["training"]["learning_rate"] = 0.002
        model = build_model((4, 1), self.config, 0)
        self.assertEqual(model.layers[0].units, 8)
        self.assertAlmostEqual(float(model.optimizer.learning_rate.numpy()), 0.002)


if __name__ == "__main__":
    unittest.main()
