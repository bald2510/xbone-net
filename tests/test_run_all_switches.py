import tempfile
import unittest
from pathlib import Path

from tools.run_all import (
    EXPERIMENTS,
    discover_experiment_configs,
    load_experiment_switches,
    select_experiments,
)


class ExperimentSwitchTests(unittest.TestCase):
    def _manifest(self, text: str):
        directory = tempfile.TemporaryDirectory()
        path = Path(directory.name) / "experiments.txt"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(directory.cleanup)
        return path

    def test_disabled_selector_wins(self):
        path = self._manifest(
            """
            [enabled]
            ctch/proposed/ours_xbone_net
            ctch/proposed/ours_xbone_net_v2
            [disabled]
            ctch/proposed/ours_xbone_net_v2
            """
        )
        self.assertEqual(
            select_experiments(None, path),
            ["ctch/proposed/ours_xbone_net"],
        )

    def test_group_prefix_is_supported(self):
        path = self._manifest(
            """
            [enabled]
            group:ctch_proposed
            [disabled]
            """
        )
        enabled, disabled = load_experiment_switches(path)
        self.assertEqual(enabled, ["ctch_proposed"])
        self.assertEqual(disabled, [])

    def test_selector_before_section_is_rejected(self):
        path = self._manifest("ctch/proposed/ours_xbone_net")
        with self.assertRaisesRegex(ValueError, "before"):
            load_experiment_switches(path)

    def test_plus_minus_syntax(self):
        path = self._manifest(
            """
            + group:ctch_proposed
            - ctch/proposed/ours_xbone_net_v2
            - ctch/proposed/proposed_v6
            """
        )
        self.assertEqual(
            select_experiments(None, path),
            ["ctch/proposed/ours_xbone_net"],
        )

    def test_empty_enabled_set_is_safe(self):
        path = self._manifest(
            """
            [enabled]
            [disabled]
            """
        )
        self.assertEqual(select_experiments(None, path), [])

    def test_registry_is_discovered_from_yaml(self):
        discovered = discover_experiment_configs()
        self.assertIn("ctch/proposed/ours_xbone_net_v2", discovered)
        self.assertIn(
            "ctch/proposed/ours_xbone_net_v2",
            EXPERIMENTS["ctch_proposed"],
        )


if __name__ == "__main__":
    unittest.main()
