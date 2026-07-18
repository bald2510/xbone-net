import unittest

import numpy as np

from src.utils.explainability import (
    curve_auc,
    rank_correlation,
    representation_quality_metrics,
    stratified_sample_indices,
)


class ExplainabilityMetricTests(unittest.TestCase):
    def test_curve_auc_and_rank_correlation(self):
        fractions = np.asarray([0.0, 0.5, 1.0])
        values = np.asarray([1.0, 0.5, 0.0])
        self.assertAlmostEqual(curve_auc(fractions, values), 0.5)
        self.assertAlmostEqual(rank_correlation(values, values), 1.0)
        self.assertAlmostEqual(rank_correlation(values, values[::-1]), -1.0)

    def test_stratified_selection_is_deterministic_and_bounded(self):
        labels = np.repeat(np.arange(4), 5)
        first = stratified_sample_indices(labels, per_class=2, seed=4)
        second = stratified_sample_indices(labels, per_class=2, seed=4)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 8)
        for class_id in range(4):
            self.assertEqual(int((labels[first] == class_id).sum()), 2)

    def test_representation_metrics_detect_separable_clusters(self):
        rng = np.random.default_rng(2)
        train = np.vstack(
            [rng.normal([-2, 0], 0.1, (30, 2)), rng.normal([2, 0], 0.1, (30, 2))]
        )
        test = np.vstack(
            [rng.normal([-2, 0], 0.1, (15, 2)), rng.normal([2, 0], 0.1, (15, 2))]
        )
        train_labels = np.repeat([0, 1], 30)
        test_labels = np.repeat([0, 1], 15)
        metrics = representation_quality_metrics(
            train, train_labels, test, test_labels, seed=1
        )
        self.assertGreater(metrics["silhouette_cosine"], 0.9)
        self.assertGreater(metrics["knn_balanced_accuracy"], 0.95)
        self.assertGreater(metrics["positive_centroid_margin_fraction"], 0.95)


if __name__ == "__main__":
    unittest.main()

