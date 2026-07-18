import unittest
from unittest.mock import patch

import torch

from src.datasets.sampling import (
    cross_class_donor_indices,
    deranged_donor_indices,
)
from src.models.builder import (
    checkpoint_model_config,
    resolve_phase_enabled,
)
from src.models.composer import XBoneMultiModalModel
from src.models.backbone.biomedclip import BiomedCLIPFoundation


class _TextOnlyBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.received_images = "unset"

    def forward(self, images, input_ids, **kwargs):
        self.received_images = images
        batch = input_ids.shape[0]
        return None, torch.ones(batch, 4)


class _DummyBiomedCLIP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = torch.nn.Linear(2, 2)
        self.text = torch.nn.Linear(2, 2)


class ExperimentControlTests(unittest.TestCase):
    def test_either_phase_gate_can_disable_execution(self):
        self.assertTrue(
            resolve_phase_enabled(
                {"run_phase1": True, "phase1": {"enabled": True}}, "phase1"
            )
        )
        self.assertFalse(
            resolve_phase_enabled(
                {"run_phase1": False, "phase1": {"enabled": True}}, "phase1"
            )
        )
        self.assertFalse(
            resolve_phase_enabled(
                {"run_phase1": True, "phase1": {"enabled": False}}, "phase1"
            )
        )

    def test_merged_checkpoint_rebuilds_without_lora(self):
        cfg = {
            "model": {
                "backbone_type": "biomedclip",
                "peft": {"type": "lora", "params": {"r": 16}},
            },
            "params": {
                "phase1": {"merge_lora_after_training": True},
                "phase2": {"init_from_phase1_merged": True},
            },
        }

        resolved = checkpoint_model_config(cfg)

        self.assertEqual(resolved["peft"], {"type": "none", "params": {}})
        self.assertEqual(cfg["model"]["peft"]["type"], "lora")

    def test_text_only_composer_does_not_call_visual_path(self):
        backbone = _TextOnlyBackbone()
        model = XBoneMultiModalModel(
            backbone=backbone,
            fusion_module=torch.nn.Identity(),
            head_module=torch.nn.Identity(),
        )
        model.use_image_in_fusion = False

        output = model(
            torch.randn(2, 3, 4, 4),
            input_ids=torch.ones(2, 3, dtype=torch.long),
        )

        self.assertIsNone(backbone.received_images)
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_report_derangements_are_one_to_one(self):
        donors = deranged_donor_indices(20, seed=7)
        self.assertEqual(len(set(donors.tolist())), 20)
        self.assertTrue(torch.as_tensor(donors != torch.arange(20).numpy()).all())

        labels = torch.tensor([0, 0, 1, 1, 2, 2]).numpy()
        donors = cross_class_donor_indices(labels, seed=9)
        self.assertTrue((labels != labels[donors]).all())

    @patch(
        "src.models.backbone.biomedclip.create_model_and_transforms",
        return_value=(_DummyBiomedCLIP(), None, None),
    )
    @patch(
        "src.models.backbone.biomedclip.get_tokenizer",
        return_value=object(),
    )
    def test_no_ft_freezes_foundation_but_keeps_resampler_trainable(
        self, _mock_tokenizer, _mock_model
    ):
        backbone = BiomedCLIPFoundation(
            freeze_base=True,
            num_visual_tokens=8,
            resampler_cfg={
                "aggregation": "passthrough",
                "use_spatial_coordinates": True,
            },
        )

        self.assertTrue(all(not p.requires_grad for p in backbone.model.parameters()))
        self.assertTrue(
            any(p.requires_grad for p in backbone.visual_resampler.parameters())
        )


if __name__ == "__main__":
    unittest.main()
