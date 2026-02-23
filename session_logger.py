"""
Session Logger for PhysioSafe VR Safety System.

Provides timestamped logging of signals and safety events.
Outputs to console and file for demo recording.

DEMO MODE: Enhanced with error handling
"""

import json
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class LogLevel(Enum):
    """Log levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


@dataclass
class LogEntry:
    """A single log entry"""
    timestamp: float
    datetime_iso: str
    level: str
    category: str
    message: str
    data: Optional[Dict] = None
    frame_number: Optional[int] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "datetime": self.datetime_iso,
            "level": self.level,
            "category": self.category,
            "message": self.message,
            "data": self.data,
            "frame": self.frame_number
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


@dataclass
class SafetyEvent:
    """A safety-related event"""
    timestamp: float
    event_type: str  # "safe", "warning", "danger", "phase_change", "correction"
    description: str
    severity: int  # 0-3
    frame_number: int
    signal_data: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "type": self.event_type,
            "description": self.description,
            "severity": self.severity,
            "frame": self.frame_number,
            "signal": self.signal_data
        }


class SessionLogger:
    """
    Session logger for PhysioSafe.
    
    Features:
    - Timestamped entries
    - Safety event tracking
    - Console and file output
    - Real-time event callbacks
    """
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        safety_events_file: Optional[str] = None,
        console_output: bool = True,
        min_log_level: LogLevel = LogLevel.INFO
    ):
        """
        Initialize session logger.
        
        Args:
            log_file: Path to log file (JSON lines format)
            safety_events_file: Path to safety events file
            console_output: Enable console output
            min_log_level: Minimum log level to record
        """
        self.log_file = log_file
        self.safety_events_file = safety_events_file
        self.console_output = console_output
        self.min_log_level = min_log_level
        
        # Data storage
        self.entries: List[LogEntry] = []
        self.safety_events: List[SafetyEvent] = []
        
        # Real-time callbacks
        self._callbacks: List[Callable[[LogEntry], None]] = []
        self._safety_callbacks: List[Callable[[SafetyEvent], None]] = []
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Statistics
        self._entry_count = 0
        self._event_count = 0
        self._start_time = time.time()
        
        # Open file handles if specified
        self._log_handle = None
        self._events_handle = None
        
        if log_file:
            try:
                self._log_handle = open(log_file, 'w')
                self._log_handle.write('[')
            except Exception as e:
                print(f"Warning: Could not open log file: {e}")
                self._log_handle = None
        
        if safety_events_file:
            try:
                self._events_handle = open(safety_events_file, 'w')
                self._events_handle.write('[\n')
            except Exception as e:
                print(f"Warning: Could not open events file: {e}")
                self._events_handle = None
    
    def log(
        self,
        message: str,
        category: str = "info",
        level: LogLevel = LogLevel.INFO,
        data: Optional[Dict] = None,
        frame_number: Optional[int] = None
    ):
        """
        Log a message.
        
        Args:
            message: Log message
            category: Category (e.g., "safety", "tracking", "system")
            level: Log level
            data: Additional data
            frame_number: Optional frame number
        """
        if level.value < self.min_log_level.value:
            return
        
        timestamp = time.time()
        entry = LogEntry(
            timestamp=timestamp,
            datetime_iso=datetime.fromtimestamp(timestamp).isoformat(),
            level=level.name,
            category=category,
            message=message,
            data=data,
            frame_number=frame_number
        )
        
        with self._lock:
            self.entries.append(entry)
            self._entry_count += 1
            
            # Write to file
            if self._log_handle:
                try:
                    if self._entry_count > 1:
                        self._log_handle.write(',\n')
                    json_line = entry.to_json()
                    self._log_handle.write(json_line)
                    self._log_handle.flush()
                except Exception as e:
                    print(f"Warning: Could not write to log file: {e}")
            
            # Call callbacks
            for callback in self._callbacks:
                try:
                    callback(entry)
                except Exception:
                    pass
        
        # Console output
        if self.console_output and level.value >= self.min_log_level.value:
            self._print_entry(entry)
    
    def log_signal(self, signal_data: Dict, frame_number: int):
        """
        Log a signal output.
        
        Args:
            signal_data: Signal dictionary
            frame_number: Frame number
        """
        safety_flag = signal_data.get('safety_flag', 'unknown')
        
        # Determine log level based on safety flag
        if safety_flag == 'danger':
            level = LogLevel.CRITICAL
        elif safety_flag == 'warning':
            level = LogLevel.WARNING
        else:
            level = LogLevel.INFO
        
        self.log(
            message=f"Signal: {safety_flag.upper()}",
            category="signal",
            level=level,
            data=signal_data,
            frame_number=frame_number
        )
    
    def log_safety_event(
        self,
        event_type: str,
        description: str,
        severity: int,
        frame_number: int,
        signal_data: Optional[Dict] = None
    ):
        """
        Log a safety event.
        
        Args:
            event_type: Event type (safe, warning, danger, phase_change, correction)
            description: Event description
            severity: Severity 0-3
            frame_number: Frame number
            signal_data: Associated signal data
        """
        timestamp = time.time()
        event = SafetyEvent(
            timestamp=timestamp,
            event_type=event_type,
            description=description,
            severity=severity,
            frame_number=frame_number,
            signal_data=signal_data
        )
        
        with self._lock:
            self.safety_events.append(event)
            self._event_count += 1
            
            # Write to file
            if self._events_handle:
                try:
                    if self._event_count > 1:
                        self._events_handle.write(',\n')
                    json_line = json.dumps(event.to_dict())
                    self._events_handle.write(json_line)
                    self._events_handle.flush()
                except Exception as e:
                    print(f"Warning: Could not write to events file: {e}")
        
        # Log the event
        if severity >= 3:
            level = LogLevel.CRITICAL
        elif severity >= 2:
            level = LogLevel.WARNING
        else:
            level = LogLevel.INFO
        
        self.log(
            message=f"EVENT: {event_type} - {description}",
            category="safety_event",
            level=level,
            data=event.to_dict(),
            frame_number=frame_number
        )
        
        # Safety callbacks
        for callback in self._safety_callbacks:
            try:
                callback(event)
            except Exception:
                pass
    
    def log_phase_change(self, old_phase: str, new_phase: str, frame_number: int):
        """Log a phase change"""
        self.log_safety_event(
            event_type="phase_change",
            description=f"Phase change: {old_phase} -> {new_phase}",
            severity=0,
            frame_number=frame_number
        )
    
    def log_correction(
        self,
        joint: str,
        direction: str,
        target: float,
        frame_number: int
    ):
        """Log a correction event"""
        self.log_safety_event(
            event_type="correction",
            description=f"Correction needed: {joint} {direction} to {target}°",
            severity=1,
            frame_number=frame_number,
            signal_data={
                "joint": joint,
                "direction": direction,
                "target_angle": target
            }
        )
    
    def add_entry_callback(self, callback: Callable[[LogEntry], None]):
        """Add callback for new log entries"""
        self._callbacks.append(callback)
    
    def add_safety_callback(self, callback: Callable[[SafetyEvent], None]):
        """Add callback for safety events"""
        self._safety_callbacks.append(callback)
    
    def get_statistics(self) -> Dict:
        """Get logger statistics"""
        return {
            "total_entries": self._entry_count,
            "safety_events": self._event_count,
            "uptime_seconds": time.time() - self._start_time,
            "log_file": self.log_file,
            "events_file": self.safety_events_file
        }
    
    def export_session(self, filepath: str):
        """Export complete session to JSON file"""
        session_data = {
            "session_info": {
                "start_time": datetime.fromtimestamp(self._start_time).isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": time.time() - self._start_time
            },
            "statistics": self.get_statistics(),
            "entries": [e.to_dict() for e in self.entries],
            "safety_events": [e.to_dict() for e in self.safety_events]
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(session_data, f, indent=2)
            print(f"Data exported to {filepath}")
        except Exception as e:
            print(f"Warning: Could not export session: {e}")
    
    def close(self):
        """Close file handles and finalize logs"""
        if self._log_handle:
            try:
                self._log_handle.write('\n]')
                self._log_handle.close()
            except Exception:
                pass
            self._log_handle = None
        
        if self._events_handle:
            try:
                self._events_handle.write('\n]')
                self._events_handle.close()
            except Exception:
                pass
            self._events_handle = None
    
    def __del__(self):
        """Ensure file handles are closed on garbage collection"""
        self.close()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False
    
    def _print_entry(self, entry: LogEntry):
        """Print entry to console"""
        timestamp_str = entry.datetime_iso.split('T')[1][:12]
        
        if entry.level == "CRITICAL":
            prefix = "❌ CRIT"
        elif entry.level == "WARNING":
            prefix = "⚠️ WARN"
        elif entry.level == "ERROR":
            prefix = "❌ ERROR"
        else:
            prefix = "✅ INFO"
        
        frame_str = f" [F{entry.frame_number}]" if entry.frame_number else ""
        
        print(f"[{timestamp_str}] {prefix}{frame_str}: {entry.message}")


# Convenience function for creating demo logger
def create_demo_logger(session_id: Optional[str] = None) -> SessionLogger:
    """
    Create a logger configured for demo recording.
    
    Args:
        session_id: Optional session identifier
        
    Returns:
        SessionLogger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_id = session_id or f"session_{timestamp}"
    
    return SessionLogger(
        log_file=f"logs/{session_id}_signals.jsonl",
        safety_events_file=f"logs/{session_id}_events.json",
        console_output=True,
        min_log_level=LogLevel.INFO
    )
