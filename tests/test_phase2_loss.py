import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.classifier.empirical_centroid import EmpiricalCentroidHead
from src.utils.losses import build_phase2_loss, resolve_phase2_loss_type


class Phase2LossTests(unittest.TestCase):
    def test_empirical_centroid_uses_weighted_cross_entropy(self):
        weights = torch.tensor([0.5, 1.5, 2.0])
        criterion = build_phase2_loss(
            "empirical_centroid_ce",
            "empirical_centroid",
            class_weights=weights,
            label_smoothing=0.0,
        )
        self.assertIsInstance(criterion, nn.CrossEntropyLoss)
        self.assertTrue(torch.equal(criterion.weight, weights))

        logits = torch.tensor([[2.0, 0.2, -0.5], [0.1, 0.4, 1.2]])
        targets = torch.tensor([0, 2])
        expected = F.cross_entropy(logits, targets, weight=weights)
        self.assertTrue(torch.allclose(criterion(logits, targets), expected))

    def test_empirical_centroid_ce_updates_features_not_centroids(self):
        head = EmpiricalCentroidHead(feature_dim=2, num_classes=2, scale=15.0)
        head.set_centroids(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([2, 2]),
        )
        features = torch.tensor([[0.6, 0.4]], requires_grad=True)
        criterion = build_phase2_loss(
            "empirical_centroid_ce", "empirical_centroid"
        )
        criterion(head(features), torch.tensor([0])).backward()

        self.assertIsNotNone(features.grad)
        self.assertGreater(float(features.grad.abs().sum()), 0.0)
        self.assertIsNone(head.centroids.grad)
        self.assertEqual(sum(p.numel() for p in head.parameters()), 0)

    def test_missing_loss_type_defaults_to_classifier_compatible_loss(self):
        self.assertEqual(
            resolve_phase2_loss_type(None, "empirical_centroid"),
            "empirical_centroid_ce",
        )
        self.assertEqual(resolve_phase2_loss_type(None, "linear"), "ce")

    def test_incompatible_loss_and_classifier_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "incompatible"):
            build_phase2_loss("ce", "empirical_centroid")
        with self.assertRaisesRegex(ValueError, "incompatible"):
            build_phase2_loss("empirical_centroid_ce", "linear")

    def test_legacy_prototype_loss_is_not_a_supported_phase2_objective(self):
        with self.assertRaisesRegex(ValueError, "classifier_type"):
            build_phase2_loss("prototypical", "prototypical")


if __name__ == "__main__":
    unittest.main()
