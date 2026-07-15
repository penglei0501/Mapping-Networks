import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from train_paper_3_1_1 import count_trainable_params
from train_paper_3_2_table4 import (
    DirectLSTMRegressor,
    MappingLSTMRegressor,
    functional_lstm_forward,
    load_npz_datasets,
    target_param_count,
)


class Table4LSTMTests(unittest.TestCase):
    def test_paper_baseline_parameter_count(self) -> None:
        model = DirectLSTMRegressor(input_size=67, hidden_size=32)
        self.assertEqual(count_trainable_params(model), 12_961)
        self.assertEqual(target_param_count(67, 32), 12_961)

    def test_functional_forward_matches_direct_lstm(self) -> None:
        torch.manual_seed(7)
        model = DirectLSTMRegressor(input_size=5, hidden_size=3)
        x = torch.randn(4, 6, 5)
        params = dict(model.named_parameters())

        expected = model(x)
        actual = functional_lstm_forward(x, params, hidden_size=3)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)

    def test_mapping_model_only_trains_latent_vector(self) -> None:
        model = MappingLSTMRegressor(
            latent_dim=64,
            input_size=5,
            hidden_size=3,
            chunk_size=32,
        )
        self.assertEqual(count_trainable_params(model), 64)
        self.assertEqual(model(torch.randn(2, 4, 5)).shape, (2,))
        self.assertFalse(model.projector.cached_W.requires_grad)

    def test_npz_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "air_pollution_sequences.npz"
            np.savez(
                path,
                x_train=np.zeros((8, 4, 67), dtype=np.float32),
                y_train=np.zeros((8,), dtype=np.float32),
                x_test=np.zeros((3, 4, 67), dtype=np.float32),
                y_test=np.zeros((3, 1), dtype=np.float32),
            )
            train_set, test_set, metadata = load_npz_datasets(path, input_size=67)

        self.assertEqual(len(train_set), 8)
        self.assertEqual(len(test_set), 3)
        self.assertEqual(metadata["x_train_shape"], [8, 4, 67])


if __name__ == "__main__":
    unittest.main()
