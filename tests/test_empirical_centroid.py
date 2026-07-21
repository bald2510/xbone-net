import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from src.models.classifier.empirical_centroid import EmpiricalCentroidHead
from src.utils.centroids import compute_empirical_centroids


class _EmbeddingDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor(
            [[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]]
        )
        self.labels = torch.tensor([0, 0, 1, 1])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "pixel_values": self.features[index],
            "labels": self.labels[index],
        }


class _IdentityEmbeddingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.head = EmpiricalCentroidHead(
            feature_dim=2, num_classes=2, scale=1.0
        )

    def encode_fused(self, images, **kwargs):
        return images


class EmpiricalCentroidTests(unittest.TestCase):
    def test_head_has_no_trainable_prototypes(self):
        head = EmpiricalCentroidHead(feature_dim=2, num_classes=2)
        self.assertEqual(list(head.parameters()), [])
        with self.assertRaises(RuntimeError):
            head(torch.ones(1, 2))

    def test_centroids_are_train_set_means(self):
        model = _IdentityEmbeddingModel()
        loader = DataLoader(_EmbeddingDataset(), batch_size=2, shuffle=False)

        counts = compute_empirical_centroids(
            model,
            loader,
            device=torch.device("cpu"),
            use_text=False,
        )

        torch.testing.assert_close(
            model.head.centroids,
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
        )
        torch.testing.assert_close(counts, torch.tensor([2, 2]))
        predictions = model.head(torch.tensor([[1.0, 0.0], [0.0, 1.0]])).argmax(1)
        torch.testing.assert_close(predictions, torch.tensor([0, 1]))

    def test_missing_class_is_rejected(self):
        head = EmpiricalCentroidHead(feature_dim=2, num_classes=2)
        with self.assertRaisesRegex(ValueError, r"Missing classes: \[1\]"):
            head.set_centroids(
                torch.tensor([[1.0, 0.0], [0.0, 0.0]]),
                torch.tensor([3, 0]),
            )

    def test_centroids_survive_checkpoint_round_trip(self):
        source = EmpiricalCentroidHead(feature_dim=2, num_classes=2)
        source.set_centroids(
            torch.tensor([[2.0, 0.0], [0.0, 3.0]]),
            torch.tensor([4, 5]),
        )
        restored = EmpiricalCentroidHead(feature_dim=2, num_classes=2)
        restored.load_state_dict(source.state_dict())

        self.assertTrue(bool(restored.centroids_initialized.item()))
        torch.testing.assert_close(restored.centroid_counts, torch.tensor([4, 5]))
        torch.testing.assert_close(restored.centroids, source.centroids)

    def test_optional_class_bias_adjusts_boundaries(self):
        head = EmpiricalCentroidHead(
            feature_dim=2,
            num_classes=2,
            scale=1.0,
            use_class_bias=True,
        )
        head.set_centroids(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([2, 2]),
        )
        with torch.no_grad():
            head.class_bias.copy_(torch.tensor([-0.5, 0.5]))

        logits = head(torch.tensor([[1.0, 0.0]]))

        torch.testing.assert_close(logits, torch.tensor([[0.5, 0.5]]))
        self.assertEqual(tuple(head.class_bias.shape), (2,))


if __name__ == "__main__":
    unittest.main()
