import unittest

import numpy as np

from hotflash_sim.vns_controller import VNSController
from hotflash_sim.vns_waveform import active_waveform_preview, generate_biphasic_waveform


class TestVNSWaveform(unittest.TestCase):
    def test_default_frequency_width_and_charge_balance(self):
        time_s, wave = generate_biphasic_waveform(0.2)
        self.assertEqual(len(time_s), 10_000)
        starts = np.flatnonzero(np.diff(wave == 1.0, prepend=False) & (wave == 1.0))
        np.testing.assert_allclose(time_s[starts], [0.0, 0.04, 0.08, 0.12, 0.16])
        for start in starts:
            np.testing.assert_array_equal(wave[start:start + 10], 1.0)
            np.testing.assert_array_equal(wave[start + 10:start + 20], -1.0)
            self.assertEqual(wave[start:start + 20].sum(), 0.0)
        self.assertEqual(np.count_nonzero(wave), 100)

    def test_interphase_gap_and_custom_amplitude(self):
        _, wave = generate_biphasic_waveform(0.01, interphase_ms=0.1, amplitude=2.5)
        expected = np.repeat([2.5, 0.0, -2.5, 0.0], [10, 5, 10, 475])
        np.testing.assert_array_equal(wave, expected)

    def test_fractional_sample_period_and_width_keep_pulses_balanced(self):
        time_s, wave = generate_biphasic_waveform(0.15, frequency_hz=27.0, pulse_width_ms=0.215)
        starts = np.flatnonzero(np.diff(wave == 1.0, prepend=False) & (wave == 1.0))
        np.testing.assert_allclose(time_s[starts], np.arange(5) / 27.0, atol=0.5 / 50_000)
        self.assertEqual(np.count_nonzero(wave == 1.0), 55)
        self.assertEqual(np.count_nonzero(wave == -1.0), 55)
        self.assertEqual(wave.sum(), 0.0)

    def test_partial_pulse_at_end_is_omitted(self):
        _, wave = generate_biphasic_waveform(0.0402)
        self.assertEqual(np.count_nonzero(wave), 20)
        self.assertEqual(wave.sum(), 0.0)
        self.assertFalse(wave[2000:].any())

    def test_impossible_settings_are_rejected(self):
        for settings in ({"frequency_hz": 0}, {"interphase_ms": -0.1}, {"sample_rate_hz": 100},
                         {"frequency_hz": 3000}, {"pulse_width_ms": float("nan")}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                generate_biphasic_waveform(0.2, **settings)

    def test_preview_is_available_only_during_controller_active_window(self):
        controller = VNSController(stimulation_s=30, cooldown_s=60)
        self.assertEqual(active_waveform_preview(controller)[1].size, 0)
        controller.on_detection({
            "session_id": "S01", "event_id": "D001", "detected_at_s": 120.25, "detected": True,
        })
        _, wave = active_waveform_preview(controller)
        self.assertEqual(np.count_nonzero(wave), 100)
        controller.update(149.75)
        self.assertEqual(active_waveform_preview(controller)[1].size, 10_000)
        controller.update(150.25)
        self.assertEqual(active_waveform_preview(controller)[1].size, 0)

    def test_short_active_window_limits_preview_and_stop_clears_it(self):
        controller = VNSController(stimulation_s=0.0625, cooldown_s=60)
        controller.on_detection({
            "session_id": "S01", "event_id": "D001", "detected_at_s": 120.25, "detected": True,
        })
        time_s, wave = active_waveform_preview(controller)
        self.assertEqual(len(time_s), 3125)
        self.assertEqual(np.count_nonzero(wave), 40)
        controller.stop(120.27)
        self.assertEqual(active_waveform_preview(controller)[1].size, 0)
