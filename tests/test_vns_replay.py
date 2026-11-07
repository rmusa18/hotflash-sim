import json
import unittest

from examples.vns_demo import baseline_detection, build_demo_replay
from hotflash_sim import OnlineFreedmanDetector
from hotflash_sim.vns_replay import build_dashboard_replay


def detection(time_s, event_id="D001"):
    return {"session_id": "S01", "event_id": event_id, "detected_at_s": time_s, "detected": True}


class TestVNSReplay(unittest.TestCase):
    def setUp(self):
        self.frames = [
            {"t_s": time_s, "eda_uS": 6.0, "pr_bpm": 72, "skin_temp_C": 33.0, "event_type": "hidden"}
            for time_s in range(7)
        ]

    def build(self, detect):
        return build_dashboard_replay(
            iter(self.frames), detect, detection_source="mock", stimulation_s=2, cooldown_s=2,
            baseline_eda_uS=2.0, assumed_effect_fraction=0.5, response_delay_s=1,
        )

    def test_replay_combines_timeline_waveform_and_aligned_scenarios(self):
        detector_inputs = []

        def detect(frame):
            detector_inputs.append(frame.copy())
            if frame["t_s"] in (2.0, 3.0):
                return detection(frame["t_s"])
            return None

        payload = json.loads(json.dumps(self.build(detect), allow_nan=False))
        self.assertEqual(payload["detection_source"], "mock")
        self.assertEqual([f["controller_state"] for f in payload["frames"]], [
            "MONITORING", "MONITORING", "STIMULATION_ACTIVE", "STIMULATION_ACTIVE",
            "COOLDOWN", "COOLDOWN", "MONITORING",
        ])
        self.assertEqual([f["active_stimulation_index"] for f in payload["frames"]],
                         [None, None, 0, 0, None, None, None])
        self.assertEqual([entry["reason"] for entry in payload["trigger_log"]], ["accepted", "duplicate"])
        self.assertEqual(len(payload["stimulations"]), 1)
        stimulation = payload["stimulations"][0]
        self.assertEqual((stimulation["event_id"], stimulation["start_s"], stimulation["end_s"],
                          stimulation["cooldown_until_s"]), ("D001", 2.0, 4.0, 6.0))
        self.assertEqual(len(stimulation["waveform_preview"]["amplitude_au"]), 10_000)
        self.assertEqual(sum(stimulation["waveform_preview"]["amplitude_au"]), 0)
        scenarios = payload["response_scenarios"]
        self.assertEqual(scenarios["no_stimulation"]["eda_uS"], [6.0] * 7)
        self.assertEqual(scenarios["zero_effect"]["eda_uS"], [6.0] * 7)
        self.assertEqual(scenarios["assumed_effect"]["eda_uS"], [6, 6, 6, 4, 4, 6, 6])
        self.assertEqual(scenarios["assumed_effect"]["signal_source"], "hypothetical")
        self.assertEqual(detector_inputs, [
            {"t_s": t, "eda_uS": 6.0, "pr_bpm": 72.0, "skin_temp_C": 33.0} for t in range(7)
        ])

    def test_missing_values_are_json_null_and_no_detection_stays_inactive(self):
        self.frames[2]["eda_uS"] = float("nan")
        self.frames[2]["pr_bpm"] = float("inf")
        self.frames[2].pop("skin_temp_C")
        payload = json.loads(json.dumps(self.build(lambda frame: None), allow_nan=False))
        self.assertEqual(payload["stimulations"], [])
        self.assertEqual(payload["frames"][2], {
            "t_s": 2.0, "eda_uS": None, "pr_bpm": None, "skin_temp_C": None,
            "controller_state": "MONITORING", "active_stimulation_index": None,
        })
        for scenario in payload["response_scenarios"].values():
            self.assertIsNone(scenario["eda_uS"][2])

    def test_future_message_is_ignored_and_cannot_advance_clock(self):
        def detect(frame):
            if frame["t_s"] in (0.0, 2.0):
                return detection(2.0)
            return None

        payload = self.build(detect)
        self.assertEqual([entry["reason"] for entry in payload["trigger_log"]],
                         ["invalid_detection", "accepted"])
        self.assertEqual(payload["frames"][0]["controller_state"], "MONITORING")
        self.assertEqual(payload["frames"][2]["controller_state"], "STIMULATION_ACTIVE")

    def test_rejects_frames_that_move_backwards(self):
        self.frames[2]["t_s"] = 0.0
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            self.build(lambda frame: None)

    def test_online_baseline_ignores_skin_warming_confirmation_after_cooldown(self):
        self.frames = [
            {"t_s": t, "eda_uS": 2.0 if t < 6 else 4.25, "pr_bpm": 72,
             "skin_temp_C": 33.0 if t < 11 else 33.5}
            for t in range(13)
        ]
        detector = OnlineFreedmanDetector(rate_hz=1)
        payload = self.build(lambda frame: baseline_detection(detector, frame))

        self.assertEqual([record["start_s"] for record in payload["stimulations"]], [6.0])
        self.assertEqual(detector.pending, [])
        self.assertEqual(payload["frames"][11]["controller_state"], "MONITORING")

    def test_seeded_baseline_demo_builds_a_labelled_end_to_end_replay(self):
        replay = json.loads(json.dumps(build_demo_replay("baseline"), allow_nan=False))
        self.assertEqual(replay["detection_source"], "nathan_online_eda_baseline")
        self.assertEqual(replay["signal_source"], "synthetic")
        self.assertEqual(replay["stimulation_mode"], "virtual_taVNS")
        self.assertIs(replay["human_stimulation"], False)
        self.assertIn("Illustrative", replay["display_labels"]["waveform"])
        self.assertIn("above baseline", replay["display_labels"]["assumed_effect_fraction"])

        self.assertEqual(len(replay["stimulations"]), 1)
        stimulation = replay["stimulations"][0]
        start_s = stimulation["start_s"]
        self.assertEqual([entry["reason"] for entry in replay["trigger_log"]], ["accepted"])
        self.assertEqual(replay["trigger_log"][0]["detected_at_s"], start_s)
        self.assertEqual(stimulation["end_s"], start_s + stimulation["duration_s"])
        self.assertEqual(stimulation["cooldown_until_s"], stimulation["end_s"] + stimulation["cooldown_s"])

        frames = {frame["t_s"]: frame for frame in replay["frames"]}
        self.assertEqual(frames[start_s]["controller_state"], "STIMULATION_ACTIVE")
        self.assertEqual(frames[stimulation["end_s"]]["controller_state"], "COOLDOWN")
        self.assertEqual(frames[stimulation["cooldown_until_s"]]["controller_state"], "MONITORING")
        self.assertEqual([t for t, frame in frames.items() if frame["active_stimulation_index"] == 0],
                         [t for t in frames if start_s <= t < stimulation["end_s"]])
        amplitude = stimulation["waveform_preview"]["amplitude_au"]
        self.assertEqual(set(amplitude), {-1.0, 0.0, 1.0})
        self.assertEqual(sum(amplitude), 0.0)

        scenarios = replay["response_scenarios"]
        original_eda = [frame["eda_uS"] for frame in frames.values()]
        self.assertEqual(scenarios["no_stimulation"]["eda_uS"], original_eda)
        self.assertEqual(scenarios["zero_effect"]["eda_uS"], original_eda)
        assumed = scenarios["assumed_effect"]
        self.assertEqual(assumed["signal_source"], "hypothetical")
        self.assertEqual(len(assumed["eda_uS"]), len(original_eda))
        self.assertNotEqual(assumed["eda_uS"], original_eda)
        response_start_s = start_s + replay["response_settings"]["response_delay_s"]
        for time_s, original, modelled in zip(frames, original_eda, assumed["eda_uS"]):
            if time_s < response_start_s:
                self.assertEqual(modelled, original)
