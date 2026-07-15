import unittest

import numpy as np

from experiments.prepare_table4_air_pollution import (
    causal_impute,
    encode_wind_direction,
    make_sequence_splits,
    parse_float,
)


class Table4AirPollutionPreparationTests(unittest.TestCase):
    def test_uci_missing_value_markers(self) -> None:
        self.assertTrue(np.isnan(parse_float("NA")))
        self.assertTrue(np.isnan(parse_float("nan")))
        self.assertTrue(np.isnan(parse_float("")))

    def test_wind_direction_encoding(self) -> None:
        sine, cosine = encode_wind_direction(
            ["N", "E", "S", "W", "", "NA", "nan"]
        )
        np.testing.assert_allclose(sine[:4], [0.0, 1.0, 0.0, -1.0], atol=1e-6)
        np.testing.assert_allclose(cosine[:4], [1.0, 0.0, -1.0, 0.0], atol=1e-6)
        self.assertTrue(np.isnan(sine[4:]).all())
        self.assertTrue(np.isnan(cosine[4:]).all())

    def test_imputation_never_backfills_from_the_future(self) -> None:
        values = np.asarray(
            [
                [np.nan, 10.0],
                [2.0, np.nan],
                [np.nan, 30.0],
                [100.0, np.nan],
            ],
            dtype=np.float32,
        )
        result, medians, missing_counts = causal_impute(values, train_end=3)

        np.testing.assert_allclose(result[:, 0], [2.0, 2.0, 2.0, 100.0])
        np.testing.assert_allclose(result[:, 1], [10.0, 10.0, 30.0, 30.0])
        np.testing.assert_allclose(medians, [2.0, 20.0])
        np.testing.assert_array_equal(missing_counts, [2, 2])

    def test_sequences_use_past_only_and_drop_missing_labels(self) -> None:
        features = np.arange(16, dtype=np.float32).reshape(8, 2)
        target = np.arange(8, dtype=np.float32)
        observed = np.asarray([True, True, True, False, True, True, True, True])
        x_train, y_train, x_test, y_test, train_indices, test_indices = (
            make_sequence_splits(
                features,
                target,
                observed,
                sequence_length=2,
                train_end=5,
            )
        )

        np.testing.assert_array_equal(train_indices, [2, 4])
        np.testing.assert_array_equal(test_indices, [5, 6, 7])
        np.testing.assert_array_equal(x_train[0], features[0:2])
        np.testing.assert_array_equal(x_test[0], features[3:5])
        np.testing.assert_array_equal(y_train, [2.0, 4.0])
        np.testing.assert_array_equal(y_test, [5.0, 6.0, 7.0])


if __name__ == "__main__":
    unittest.main()
