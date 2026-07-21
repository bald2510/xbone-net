import unittest

import torch
from torch.utils.data import Dataset

from src.utils.trainer import ClassAwareSampler


class _LabelDataset(Dataset):
    def __init__(self):
        self.labels = torch.arange(48) % 6

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return index


class ClassAwareSamplerTests(unittest.TestCase):
    def test_full_batches_contain_same_class_pairs(self):
        dataset = _LabelDataset()
        sampler = ClassAwareSampler(
            dataset,
            batch_size=12,
            samples_per_class=2,
            seed=7,
        )

        indices = list(iter(sampler))

        self.assertEqual(len(indices), len(dataset))
        for start in range(0, len(indices), 12):
            labels = dataset.labels[indices[start:start + 12]]
            counts = torch.bincount(labels, minlength=6)
            self.assertTrue(torch.all((counts == 0) | (counts >= 2)))

    def test_epoch_changes_order_deterministically(self):
        dataset = _LabelDataset()
        sampler = ClassAwareSampler(dataset, batch_size=12, seed=11)
        first = list(iter(sampler))
        sampler.set_epoch(1)
        second = list(iter(sampler))
        sampler.set_epoch(0)

        self.assertNotEqual(first, second)
        self.assertEqual(first, list(iter(sampler)))


if __name__ == "__main__":
    unittest.main()
