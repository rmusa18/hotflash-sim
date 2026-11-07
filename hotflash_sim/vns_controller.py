"""Schedule virtual stimulation using seconds from the start of a simulation."""

from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from math import isfinite


class StimulationState(str, Enum):
    MONITORING = "MONITORING"
    STIMULATION_ACTIVE = "STIMULATION_ACTIVE"
    COOLDOWN = "COOLDOWN"


@dataclass(frozen=True)
class Stimulation:
    session_id: str
    event_id: str
    start_s: float
    end_s: float
    cooldown_until_s: float


@dataclass(frozen=True)
class TriggerDecision:
    session_id: str | None
    event_id: str | None
    detected_at_s: float | None
    reason: str


def _seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        seconds = float(value)
    except OverflowError:
        return None
    return seconds if isfinite(seconds) and seconds >= 0 else None


class VNSController:
    """Run one replay clock, with configurable stimulation and cooldown durations.

    Call update() with each frame's timestamp and on_detection() with detector
    events in timestamp order. Reset before replaying a different session.
    """

    def __init__(self, *, stimulation_s: float, cooldown_s: float) -> None:
        duration = _seconds(stimulation_s)
        cooldown = _seconds(cooldown_s)
        if duration is None or duration == 0 or cooldown is None:
            raise ValueError("Stimulation must be positive and cooldown nonnegative, in seconds.")
        self.stimulation_s = duration
        self.cooldown_s = cooldown
        self.stimulations: list[Stimulation] = []
        self.trigger_log: list[TriggerDecision] = []
        self._seen: set[tuple[str, str]] = set()
        self._time_s = 0.0

    @property
    def state(self) -> StimulationState:
        if self.stimulations:
            latest = self.stimulations[-1]
            if self._time_s < latest.end_s:
                return StimulationState.STIMULATION_ACTIVE
            if self._time_s < latest.cooldown_until_s:
                return StimulationState.COOLDOWN
        return StimulationState.MONITORING

    def update(self, time_s: object) -> StimulationState:
        """Advance to a valid frame timestamp. Ignore missing or older timestamps."""
        timestamp = _seconds(time_s)
        if timestamp is not None and timestamp >= self._time_s:
            self._time_s = timestamp
        return self.state

    def on_detection(self, event: object) -> Stimulation | None:
        """Accept the proposed detection message and log the decision."""
        payload = event if isinstance(event, Mapping) else {}
        session = payload.get("session_id")
        event_id = payload.get("event_id")
        session = session if isinstance(session, str) and session.strip() else None
        event_id = event_id if isinstance(event_id, str) and event_id.strip() else None
        timestamp = _seconds(payload.get("detected_at_s"))
        stimulation = None

        if session is None or event_id is None or timestamp is None or payload.get("detected") is not True:
            reason = "invalid_detection"
        elif timestamp < self._time_s:
            reason = "stale_detection"
        else:
            state = self.update(timestamp)
            identity = (session, event_id)
            if identity in self._seen:
                reason = "duplicate"
            elif state is StimulationState.STIMULATION_ACTIVE:
                reason = "stimulation_active"
            elif state is StimulationState.COOLDOWN:
                reason = "cooldown"
            else:
                end_s = timestamp + self.stimulation_s
                stimulation = Stimulation(session, event_id, timestamp, end_s, end_s + self.cooldown_s)
                self.stimulations.append(stimulation)
                reason = "accepted"
            self._seen.add(identity)

        self.trigger_log.append(TriggerDecision(session, event_id, timestamp, reason))
        return stimulation

    def stop(self, time_s: object) -> StimulationState:
        """End an active window early and begin cooldown at the stop timestamp."""
        timestamp = _seconds(time_s)
        if timestamp is None or timestamp < self._time_s:
            return self.state
        if self.update(timestamp) is StimulationState.STIMULATION_ACTIVE:
            self.stimulations[-1] = replace(
                self.stimulations[-1], end_s=timestamp, cooldown_until_s=timestamp + self.cooldown_s
            )
        return self.state

    def reset(self) -> None:
        """Clear the clock, stimulation history and trigger log for a new replay."""
        self._time_s = 0.0
        self.stimulations.clear()
        self.trigger_log.clear()
        self._seen.clear()
