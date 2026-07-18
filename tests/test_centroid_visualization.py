import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.utils.centroid_visualization import (
    compute_centroid_diagnostics,
    load_centroids_from_checkpoint,
    project_cosine_space,
    stratified_subsample,
)


class CentroidVisualizationTests(unittest.TestCase):
    def test_loads_prefixed_centroid_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "model.pth"
            torch.save(
                {
                    "module.head.centroids": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                    "module.head.centroid_counts": torch.tensor([3, 4]),
                    "module.head.centroids_initialized": torch.tensor(True),
                },
                checkpoint,
            )

            centroids, counts = load_centroids_from_checkpoint(checkpoint)

        np.testing.assert_allclose(centroids, [[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_equal(counts, [3, 4])

    def test_diagnostics_detect_margin_and_nearest_centroid(self):
        centroids = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        embeddings = np.asarray([[1.0, 0.1], [0.1, 1.0]], dtype=np.float32)
        labels = np.asarray([0, 1])

        diagnostics = compute_centroid_diagnostics(centroids, embeddings, labels)

        np.testing.assert_array_equal(diagnostics["nearest_centroid_ids"], [1, 0])
        self.assertEqual(diagnostics["overall_nearest_centroid_accuracy"], 1.0)
        self.assertGreater(diagnostics["overall_mean_cosine_margin"], 0.0)

    def test_projection_returns_centroids_and_samples(self):
        centroids = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        embeddings = np.asarray([[1.0, 0.1, 0.0], [0.1, 1.0, 0.0]])

        projected_centroids, projected_samples, explained = project_cosine_space(
            centroids, embeddings
        )

        self.assertEqual(projected_centroids.shape, (2, 2))
        self.assertEqual(projected_samples.shape, (2, 2))
        self.assertEqual(explained.shape, (2,))

    def test_subsampling_is_stratified_and_deterministic(self):
        embeddings = np.arange(40, dtype=np.float32).reshape(20, 2)
        labels = np.repeat([0, 1], 10)

        first = stratified_subsample(embeddings, labels, 2, 3, seed=42)
        second = stratified_subsample(embeddings, labels, 2, 3, seed=42)

        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(np.bincount(first[1]), [3, 3])


if __name__ == "__main__":
    unittest.main()
