import unittest

from hotflash_sim.vns_controller import StimulationState, VNSController


def detection(event_id="D001", time_s=120.0):
    return {
        "session_id": "S01",
        "event_id": event_id,
        "detected_at_s": time_s,
        "detected": True,
        "model": "eda_threshold",
        "score": None,
    }


class TestVNSController(unittest.TestCase):
    def setUp(self):
        self.controller = VNSController(stimulation_s=30.0, cooldown_s=60.0)

    def test_no_detection_leaves_controller_monitoring(self):
        self.assertIs(self.controller.update(119.5), StimulationState.MONITORING)
        self.assertEqual(self.controller.stimulations, [])

    def test_one_event_starts_and_ends_at_exact_simulation_times(self):
        controller = self.controller
        controller.update(120.0)
        stimulation = controller.on_detection(detection())
        self.assertEqual((stimulation.start_s, stimulation.end_s, stimulation.cooldown_until_s),
                         (120.0, 150.0, 210.0))
        self.assertIs(controller.update(149.5), StimulationState.STIMULATION_ACTIVE)
        self.assertIs(controller.update(150.0), StimulationState.COOLDOWN)
        self.assertIs(controller.update(209.5), StimulationState.COOLDOWN)
        self.assertIs(controller.update(210.0), StimulationState.MONITORING)
        self.assertEqual(len(controller.stimulations), 1)

    def test_duplicate_and_lockout_decisions_are_logged(self):
        controller = self.controller
        controller.on_detection(detection())
        self.assertIsNone(controller.on_detection(detection()))
        self.assertIsNone(controller.on_detection(detection("D002", 149.5)))
        self.assertIsNone(controller.on_detection(detection("D003", 150.0)))
        self.assertIsNone(controller.on_detection(detection("D003", 210.0)))
        self.assertIsNotNone(controller.on_detection(detection("D004", 210.0)))
        self.assertEqual([entry.reason for entry in controller.trigger_log],
                         ["accepted", "duplicate", "stimulation_active", "cooldown", "duplicate", "accepted"])

    def test_stop_starts_cooldown_at_early_stop_time(self):
        controller = self.controller
        controller.on_detection(detection())
        self.assertIs(controller.stop(130.0), StimulationState.COOLDOWN)
        self.assertIs(controller.stop(135.0), StimulationState.COOLDOWN)
        self.assertEqual(controller.stimulations[0].end_s, 130.0)
        self.assertIs(controller.update(190.0), StimulationState.MONITORING)

    def test_reset_allows_same_session_to_replay(self):
        controller = self.controller
        controller.on_detection(detection())
        controller.update(220.0)
        controller.reset()
        self.assertIs(controller.state, StimulationState.MONITORING)
        self.assertEqual(controller.stimulations, [])
        self.assertEqual(controller.trigger_log, [])
        self.assertIsNotNone(controller.on_detection(detection()))

    def test_invalid_and_stale_events_do_not_activate(self):
        controller = self.controller
        controller.update(120.5)
        for event in (None, {}, detection(time_s=float("nan")), detection(time_s=True),
                      {**detection(time_s=999.0), "detected": False}, detection()):
            with self.subTest(event=event):
                self.assertIsNone(controller.on_detection(event))
                self.assertIs(controller.state, StimulationState.MONITORING)
        self.assertEqual(controller.trigger_log[-1].reason, "stale_detection")
        self.assertIsNotNone(controller.on_detection(detection(time_s=120.5)))
        self.assertIs(controller.update(None), StimulationState.STIMULATION_ACTIVE)
        self.assertIs(controller.update(120.0), StimulationState.STIMULATION_ACTIVE)

    def test_replay_can_jump_past_both_deadlines(self):
        controller = self.controller
        controller.on_detection(detection())
        self.assertIs(controller.update(500.0), StimulationState.MONITORING)
        self.assertEqual(controller.stimulations[0].end_s, 150.0)
        self.assertIsNotNone(controller.on_detection(detection("D002", 500.0)))
