import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_ood import (
    SCENARIO_FEATURES,
    _validate_archive,
    align_paired_id,
    run_protocol,
)
from src.datasets.fracatlas import FRACATLAS_IMAGE_RESOLVER_VERSION
from src.utils.analysis import (
    SOURCE_EXPERIMENT,
    load_feature_archive,
    save_feature_archive,
)
from src.utils.ood import (
    MAHALANOBIS_SCORE_DEFINITION,
    OODDetector,
    bootstrap_ood_metrics,
    calibrate_ood_threshold,
    evaluate_ood,
)
from tools.benchmark.ood_analysis import OOD_FEATURE_KEYS


class OODProtocolTests(unittest.TestCase):
    def test_domain_ood_defaults_to_trained_multimodal_fusion(self):
        self.assertEqual(SCENARIO_FEATURES["domain_ood"], "fused_embeddings")
        self.assertEqual(
            SCENARIO_FEATURES["domain_ood_btxrd"], "fused_embeddings"
        )
        self.assertEqual(set(OOD_FEATURE_KEYS.values()), {"fused_embeddings"})

    def test_semantic_and_report_mismatch_use_multimodal_fusion(self):
        scenarios = (
            "semantic_ood",
            "report_mismatch_cross_class",
            "report_mismatch_same_class",
        )
        for scenario in scenarios:
            with self.subTest(scenario=scenario):
                self.assertEqual(
                    SCENARIO_FEATURES[scenario], "fused_embeddings"
                )
                self.assertEqual(
                    OOD_FEATURE_KEYS[scenario], "fused_embeddings"
                )

    def test_scores_use_higher_as_more_ood(self):
        logits_id = np.asarray([[8.0, 0.0], [0.0, 7.0]])
        logits_ood = np.asarray([[0.1, 0.0], [0.0, 0.1]])
        detector = OODDetector()
        for scorer in (
            detector.score_msp,
            detector.score_entropy,
            detector.score_energy,
            detector.score_max_logit,
        ):
            self.assertGreater(scorer(logits_ood).mean(), scorer(logits_id).mean())

    def test_mahalanobis_matches_classical_distance_formula(self):
        train = np.asarray(
            [[1.0, 1.0], [3.0, 1.0], [1.0, 3.0], [3.0, 3.0]]
        )
        detector = OODDetector().fit(train, np.zeros(len(train), dtype=np.int64))
        probe = np.asarray([[4.0, 5.0]])
        center = detector.class_means[0]
        difference = probe[0] - center
        expected = np.sqrt(
            difference @ detector.shared_cov_inv @ difference
        )

        np.testing.assert_allclose(center, [2.0, 2.0])
        self.assertAlmostEqual(
            float(detector.score_mahalanobis(probe)[0]), float(expected)
        )

    def test_threshold_depends_only_on_calibration_id(self):
        calibration = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5])
        first = calibrate_ood_threshold(calibration, target_id_fpr=0.2)
        _ = evaluate_ood(calibration, np.asarray([1.0, 2.0]), first)
        second = calibrate_ood_threshold(calibration, target_id_fpr=0.2)
        _ = evaluate_ood(calibration, np.asarray([100.0, 200.0]), second)
        self.assertEqual(first, second)

    def test_fpr95_uses_standard_id_tpr_definition(self):
        # One extreme ID outlier forces the 95%-ID-acceptance threshold above
        # every OOD score. The reverse OOD-positive operating point is smaller.
        id_scores = np.asarray([0.0] * 18 + [10.0, 10.0])
        ood_scores = np.linspace(0.5, 1.5, 20)
        metrics = evaluate_ood(id_scores, ood_scores)
        self.assertEqual(
            metrics["fpr95_definition"],
            "ood_accepted_as_id_at_95_percent_id_tpr",
        )
        self.assertGreater(metrics["fpr_at_95tpr"], 0.9)
        self.assertLess(metrics["id_fpr_at_95_ood_tpr"], 0.2)

    def test_paired_bootstrap_is_deterministic(self):
        id_scores = np.linspace(0.0, 0.4, 20)
        ood_scores = id_scores + 0.5
        first = bootstrap_ood_metrics(
            id_scores, ood_scores, n_bootstrap=30, seed=7, paired=True
        )
        second = bootstrap_ood_metrics(
            id_scores, ood_scores, n_bootstrap=30, seed=7, paired=True
        )
        self.assertEqual(first, second)

    def test_align_report_mismatch_by_image_id(self):
        native = {
            "labels": np.asarray([0, 1, 2]),
            "image_id": np.asarray(["a", "b", "c"]),
            "logits": np.eye(3),
        }
        mismatch = {
            "labels": np.asarray([2, 0]),
            "image_id": np.asarray(["c", "a"]),
            "logits": np.zeros((2, 3)),
        }
        aligned = align_paired_id(native, mismatch)
        np.testing.assert_array_equal(aligned["image_id"], ["c", "a"])
        np.testing.assert_array_equal(aligned["labels"], [2, 0])

    def test_domain_archive_requires_complete_nonconstant_visual_inputs(self):
        arrays = {
            "labels": np.asarray([0, 1]),
            "logits": np.zeros((2, 2)),
            "image_id": np.asarray(["a", "b"]),
            "visual_global_embeddings": np.asarray(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
        }
        base = {
            "source_experiment": SOURCE_EXPERIMENT,
            "seed": 42,
            "scenario": "fracatlas_test",
        }
        with self.assertRaises(ValueError):
            _validate_archive(arrays, base, 42, "fracatlas_test")

        complete = {
            **base,
            "fracatlas_image_resolver_version": (
                FRACATLAS_IMAGE_RESOLVER_VERSION
            ),
            "coverage": {
                "rows": 2,
                "resolved_images": 2,
                "missing_images": 0,
                "missing_reports": 0,
                "decode_validation": True,
                "decode_failure_count": 0,
            },
        }
        _validate_archive(arrays, complete, 42, "fracatlas_test")

        constant = {
            **arrays,
            "visual_global_embeddings": np.ones((2, 2)),
        }
        with self.assertRaises(ValueError):
            _validate_archive(constant, complete, 42, "fracatlas_test")

    def test_btxrd_domain_archive_requires_complete_coverage(self):
        arrays = {
            "labels": np.asarray([0, 1]),
            "logits": np.zeros((2, 22)),
            "image_id": np.asarray(["a", "b"]),
            "visual_global_embeddings": np.asarray(
                [[1.0, 0.0], [0.0, 1.0]]
            ),
        }
        provenance = {
            "source_experiment": SOURCE_EXPERIMENT,
            "seed": 42,
            "scenario": "btxrd_test",
            "ood_dataset": "BTXRD",
            "ood_split": "test",
            "coverage": {
                "rows": 2,
                "resolved_images": 2,
                "missing_images": 0,
                "missing_xray_reports": 0,
                "missing_clinical_reports": 0,
            },
        }
        _validate_archive(arrays, provenance, 42, "btxrd_test")

        incomplete = {
            **provenance,
            "coverage": {
                **provenance["coverage"],
                "missing_clinical_reports": 1,
            },
        }
        with self.assertRaises(ValueError):
            _validate_archive(arrays, incomplete, 42, "btxrd_test")

    def test_full_protocol_uses_disjoint_inputs(self):
        rng = np.random.default_rng(3)
        train_labels = np.repeat([0, 1], 20)
        train_features = np.vstack(
            [rng.normal([-1, 0], 0.05, (20, 2)), rng.normal([1, 0], 0.05, (20, 2))]
        )

        def archive(features, labels, logits=None):
            if logits is None:
                logits = np.column_stack([-features[:, 0], features[:, 0]]) * 5
            return {
                "fused_embeddings": features,
                "labels": np.asarray(labels),
                "logits": logits,
                "image_id": np.asarray([f"i{index}" for index in range(len(labels))]),
                "group": np.asarray(["all"] * len(labels)),
            }

        train = archive(train_features, train_labels)
        calibration = archive(train_features[:10], train_labels[:10])
        id_test = archive(train_features[20:30], train_labels[20:30])
        ood_features = rng.normal([0, 2], 0.05, (10, 2))
        ood = archive(ood_features, np.full(10, -1), np.zeros((10, 2)))
        results, scores = run_protocol(
            train,
            calibration,
            id_test,
            ood,
            methods=["mahalanobis", "msp"],
            feature_key="fused_embeddings",
            target_id_fpr=0.05,
            knn_k=3,
            temperature=1.0,
            n_bootstrap=20,
            bootstrap_seed=11,
            paired=False,
        )
        self.assertIn("mahalanobis", results)
        self.assertEqual(
            results["mahalanobis"]["score_definition"],
            MAHALANOBIS_SCORE_DEFINITION,
        )
        self.assertEqual(len(scores["mahalanobis_calibration"]), len(calibration["labels"]))
        self.assertGreater(results["mahalanobis"]["auroc"], 0.9)

    def test_feature_archive_requires_locked_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "features.npz"
            save_feature_archive(
                path,
                {"labels": np.asarray([0]), "image_id": np.asarray(["a"])},
                {"source_experiment": SOURCE_EXPERIMENT, "seed": 42},
            )
            arrays, provenance = load_feature_archive(path)
            self.assertEqual(provenance["source_experiment"], SOURCE_EXPERIMENT)
            np.testing.assert_array_equal(arrays["labels"], [0])


if __name__ == "__main__":
    unittest.main()
