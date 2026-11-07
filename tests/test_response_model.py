import unittest

import numpy as np
import pandas as pd

from hotflash_sim.response_model import simulate_response_scenarios
from hotflash_sim.vns_controller import VNSController


class TestResponseModel(unittest.TestCase):
    def setUp(self):
        self.signals = pd.DataFrame({
            "t_s": np.arange(11, dtype=float),
            "eda_uS": [6.0] * 11,
            "pr_bpm": np.arange(60, 71),
            "skin_temp_C": np.linspace(32.0, 33.0, 11),
        })
        self.controller = VNSController(stimulation_s=4, cooldown_s=0)
        self.controller.on_detection({
            "session_id": "S01", "event_id": "D001", "detected_at_s": 2.0, "detected": True,
        })

    def compare(self, effect=0.5, **timing):
        return simulate_response_scenarios(
            self.signals, self.controller.stimulations,
            baseline_eda_uS=2.0, assumed_effect_fraction=effect, **timing,
        )

    def test_zero_strength_preserves_all_trajectories_and_labels_stimulation(self):
        scenarios = self.compare(effect=0.0, response_delay_s=1)
        for scenario in scenarios.values():
            pd.testing.assert_frame_equal(scenario.signals, self.signals, check_exact=True)
        self.assertEqual(scenarios["no_stimulation"].stimulations, ())
        self.assertEqual(scenarios["zero_effect"].stimulations, tuple(self.controller.stimulations))

    def test_assumed_response_builds_after_delay_and_recovers(self):
        scenarios = self.compare(response_delay_s=1, development_s=2, recovery_s=2)
        assumed = scenarios["assumed_effect"]
        np.testing.assert_array_equal(assumed.signals["eda_uS"], [6, 6, 6, 6, 5, 4, 4, 4, 5, 6, 6])
        pd.testing.assert_frame_equal(assumed.signals.drop(columns="eda_uS"),
                                      self.signals.drop(columns="eda_uS"), check_exact=True)
        pd.testing.assert_frame_equal(scenarios["zero_effect"].signals, self.signals, check_exact=True)
        self.assertEqual(assumed.assumed_effect_fraction, 0.5)

    def test_early_stop_limits_build_up(self):
        self.controller.stop(3.0)
        scenarios = self.compare(response_delay_s=1, development_s=4, recovery_s=2)
        np.testing.assert_array_equal(scenarios["assumed_effect"].signals["eda_uS"],
                                      [6, 6, 6, 6, 5.5, 5.75, 6, 6, 6, 6, 6])

    def test_aligned_baseline_preserves_missing_samples_and_values_below_baseline(self):
        self.signals.loc[3, "eda_uS"] = np.nan
        self.signals.loc[4, "eda_uS"] = 1.0
        baseline = np.linspace(2.0, 3.0, 11)
        scenarios = simulate_response_scenarios(
            self.signals, self.controller.stimulations, baseline_eda_uS=baseline, assumed_effect_fraction=1.0,
        )
        eda = scenarios["assumed_effect"].signals["eda_uS"]
        np.testing.assert_array_equal(eda.iloc[2:7], [2.2, np.nan, 1.0, 2.5, 6.0])
        self.assertEqual(self.signals.loc[2, "eda_uS"], 6.0)

    def test_overlapping_recovery_uses_strongest_effect_without_stacking(self):
        self.controller.on_detection({
            "session_id": "S01", "event_id": "D002", "detected_at_s": 6.0, "detected": True,
        })
        eda = self.compare(recovery_s=4)["assumed_effect"].signals["eda_uS"]
        np.testing.assert_array_equal(eda.iloc[5:10], [4, 4, 4, 4, 4])

    def test_no_stimulation_leaves_assumed_scenario_unchanged(self):
        self.controller.reset()
        assumed = self.compare()["assumed_effect"]
        pd.testing.assert_frame_equal(assumed.signals, self.signals, check_exact=True)

    def test_rejects_invalid_parameters(self):
        for options in ({"assumed_effect_fraction": 1.1}, {"response_delay_s": -1},
                        {"recovery_s": float("nan")}, {"baseline_eda_uS": [1, 2]}):
            settings = {"baseline_eda_uS": 2.0, "assumed_effect_fraction": 0.5, **options}
            with self.subTest(options=options), self.assertRaises(ValueError):
                simulate_response_scenarios(self.signals, self.controller.stimulations, **settings)
