"""
Safety Event Manager

Provides structured runtime event state handling over the per-frame safety assessments/signals.
- Tracks current safety state (safe/warning/danger/unknown)
- Emits well-formed transition events (enter/exit/escalate/deescalate)
- Debounces noisy transitions (min dwell time)
- Maintains a bounded event history for consumption by APIs/streams

Usage:
  mgr = SafetyEventManager()
  events = mgr.push_frame(assessment, signal)
  # events is a list of emitted events (possibly empty)

Event shape:
{
  "event_id": str,
  "type": "enter"|"exit"|"escalate"|"deescalate"|"heartbeat",
  "from": "safe|warning|danger|unknown",
  "to": "safe|warning|danger|unknown",
  "level": "safe|warning|danger|unknown",
  "frame": int,
  "timestamp": float,
  "violations": int,
  "primary_violation": Optional[str],
}
"""

import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from safety_rules import SafetyAssessment, SafetyLevel
from signal_generator import SafetyFrameSignal


@dataclass
class Event:
    event_id: str
    type: str
    from_level: str
    to_level: str
    level: str
    frame: int
    timestamp: float
    violations: int
    primary_violation: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "from": self.from_level,
            "to": self.to_level,
            "level": self.level,
            "frame": self.frame,
            "timestamp": self.timestamp,
            "violations": self.violations,
            "primary_violation": self.primary_violation,
        }


@dataclass
class SafetyEventManager:
    min_dwell_seconds: float = 0.3  # debounce dwell time before confirming transitions
    history_size: int = 200
    _current_level: str = field(default="unknown", init=False)
    _candidate_level: Optional[str] = field(default=None, init=False)
    _candidate_since: float = field(default=0.0, init=False)
    _last_heartbeat: float = field(default=0.0, init=False)
    _history: Deque[Event] = field(default_factory=lambda: deque(maxlen=200), init=False)

    def _emit(self, ev: Event) -> None:
        self._history.append(ev)

    def _now(self) -> float:
        return time.time()

    @staticmethod
    def _lvl_value(level: str) -> int:
        order = {"unknown": -1, "safe": 0, "warning": 1, "danger": 2}
        return order.get(level, -1)

    def _classify_transition(self, from_l: str, to_l: str) -> str:
        if from_l == to_l:
            return "heartbeat"
        if self._lvl_value(to_l) > self._lvl_value(from_l):
            # upward severity
            return "escalate" if from_l != "unknown" else "enter"
        # downward or return to safe
        return "deescalate" if to_l != "unknown" and from_l != "unknown" else "exit"

    def push_frame(self, assessment: SafetyAssessment, signal: SafetyFrameSignal) -> List[Dict]:
        """Ingest a frame and emit any transition events."""
        level = assessment.overall_safety.value
        ts = signal.timestamp if signal else assessment.timestamp
        frame = signal.frame_number if signal else assessment.frame_number
        events: List[Dict] = []

        # Debounce: if level changes, start candidate; confirm if dwell time met
        if level != self._current_level:
            # new candidate or same candidate continues
            if self._candidate_level != level:
                self._candidate_level = level
                self._candidate_since = self._now()
            # If dwell exceeded, confirm transition
            if (self._now() - self._candidate_since) >= self.min_dwell_seconds:
                ev_type = self._classify_transition(self._current_level, level)
                ev = Event(
                    event_id=str(uuid.uuid4()),
                    type=ev_type,
                    from_level=self._current_level,
                    to_level=level,
                    level=level,
                    frame=frame,
                    timestamp=ts,
                    violations=signal.active_violations if signal else len(assessment.violations),
                    primary_violation=signal.primary_violation if signal else (
                        f"{assessment.violations[0].joint} {assessment.violations[0].movement}" if assessment.violations else None
                    ),
                )
                self._emit(ev)
                events.append(ev.to_dict())
                self._current_level = level
                self._candidate_level = None
                self._candidate_since = 0.0
        else:
            # Regular heartbeat every ~1s while stable
            now = self._now()
            if now - self._last_heartbeat > 1.0:
                hb = Event(
                    event_id=str(uuid.uuid4()),
                    type="heartbeat",
                    from_level=self._current_level,
                    to_level=self._current_level,
                    level=self._current_level,
                    frame=frame,
                    timestamp=ts,
                    violations=signal.active_violations if signal else len(assessment.violations),
                    primary_violation=signal.primary_violation if signal else (
                        f"{assessment.violations[0].joint} {assessment.violations[0].movement}" if assessment.violations else None
                    ),
                )
                self._emit(hb)
                events.append(hb.to_dict())
                self._last_heartbeat = now

        return events

    def history(self, limit: Optional[int] = None) -> List[Dict]:
        items = list(self._history)[-limit:] if limit else list(self._history)
        return [e.to_dict() for e in items]
