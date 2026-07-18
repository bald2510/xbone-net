import unittest

import numpy as np

from src.utils.metrics import multiclass_calibration_metrics


class CalibrationMetricTests(unittest.TestCase):
    def test_perfect_confident_predictions_have_small_calibration_loss(self):
        probabilities = np.asarray(
            [[0.999, 0.001], [0.001, 0.999], [0.999, 0.001], [0.001, 0.999]]
        )
        labels = np.asarray([0, 1, 0, 1])

        metrics = multiclass_calibration_metrics(probabilities, labels)

        self.assertLess(metrics["ece_15"], 0.01)
        self.assertLess(metrics["adaptive_ece_15"], 0.01)
        self.assertLess(metrics["nll"], 0.01)
        self.assertLess(metrics["brier_score"], 0.01)

    def test_confident_wrong_predictions_are_penalized(self):
        probabilities = np.asarray([[0.99, 0.01], [0.01, 0.99]])
        labels = np.asarray([1, 0])

        metrics = multiclass_calibration_metrics(probabilities, labels)

        self.assertGreater(metrics["ece_15"], 0.9)
        self.assertGreater(metrics["nll"], 4.0)
        self.assertGreater(metrics["brier_score"], 1.5)


if __name__ == "__main__":
    unittest.main()
