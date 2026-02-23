"""
Neuro-Safe Real-Time Signal Engine for physiotherapy safety system.
Emits clean, unambiguous per-frame JSON signals with:
- phase: Exercise phase detection
- severity: Urgency level (0-3)
- correction: Specific guidance for correction
- confidence: Assessment certainty (0-1)
- safety_flag: Overall safety status
- Cooldowns: Prevent feedback spam
- De-duplication: Suppress repeated signals

DEMO MODE: Enhanced for stability and clarity
"""

import json
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
from safety_rules import SafetyAssessment, SafetyLevel


class SignalType(Enum):
    """Types of output signals"""
    STATUS = "status"
    WARNING = "warning"
    DANGER = "danger"
    COMMAND = "command"
    DATA = "data"


class ExercisePhase(Enum):
    """Exercise phase detection"""
    REST = "rest"
    INITIATION = "initiation"
    ACTIVE = "active"
    TRANSITION = "transition"
    COMPLETION = "completion"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Signal severity levels (0-3, higher = more severe)"""
    INFO = 0      # Informational
    LOW = 1       # Minor issue
    MEDIUM = 2    # Requires attention
    HIGH = 3     # Critical - immediate action


@dataclass
class CorrectionGuidance:
    """Correction guidance for unsafe positions"""
    joint: str
    movement: str
    direction: str  # "raise", "lower", "straighten", "bend"
    target_angle: Optional[float]
    instruction: str
    
    def to_dict(self) -> Dict:
        return {
            "joint": self.joint,
            "movement": self.movement,
            "direction": self.direction,
            "target_angle": round(self.target_angle, 1) if self.target_angle else None,
            "instruction": self.instruction
        }


@dataclass
class SafetyFrameSignal:
    """
    Per-frame safety signal with all required fields.
    This is the core output format for the neuro-safe engine.
    """
    # Core fields
    frame_number: int
    timestamp: float
    
    # Safety assessment
    safety_flag: str           # "safe", "warning", "danger", "unknown"
    confidence: float          # 0.0 - 1.0
    
    # Severity (0-3)
    severity: int
    
    # Exercise phase
    phase: str                # rest, initiation, active, transition, completion
    
    # Correction guidance (None if safe)
    correction: Optional[Dict] = None
    
    # Signal deduplication
    is_new: bool = True       # True if new signal, False if suppressed
    signal_code: str = ""      # For deduplication
    
    # Additional context
    active_violations: int = 0
    primary_violation: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "frame": self.frame_number,
            "timestamp": round(self.timestamp, 4),
            "safety_flag": self.safety_flag,
            "confidence": round(self.confidence, 3),
            "severity": self.severity,
            "phase": self.phase,
            "correction": self.correction,
            "is_new": self.is_new,
            "signal_code": self.signal_code,
            "active_violations": self.active_violations,
            "primary_violation": self.primary_violation
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SignalStatistics:
    """Statistics for signal generation"""
    total_frames: int = 0
    safe_signals: int = 0
    warning_signals: int = 0
    danger_signals: int = 0
    suppressed_signals: int = 0
    phase_changes: int = 0
    last_signal_time: float = 0
    
    def to_dict(self) -> Dict:
        return {
            "total_frames": self.total_frames,
            "safe": self.safe_signals,
            "warning": self.warning_signals,
            "danger": self.danger_signals,
            "suppressed": self.suppressed_signals,
            "phase_changes": self.phase_changes
        }


class NeuroSafeSignalEngine:
    """
    Neuro-safe real-time signal engine.
    
    Features:
    - Per-frame JSON signals with phase, severity, correction, confidence, safety_flag
    - Cooldown system to prevent feedback spam
    - De-duplication of repeated signals
    - Deterministic output (same input = same output)
    """
    
    # Severity mapping
    SEVERITY_MAP = {
        SafetyLevel.SAFE: Severity.INFO,
        SafetyLevel.WARNING: Severity.MEDIUM,
        SafetyLevel.DANGER: Severity.HIGH,
        SafetyLevel.UNKNOWN: Severity.LOW
    }
    
    # Cooldown periods (seconds)
    COOLDOWN_MAP = {
        SafetyLevel.SAFE: 1.0,
        SafetyLevel.WARNING: 0.5,
        SafetyLevel.DANGER: 0.0,  # No cooldown for danger
        SafetyLevel.UNKNOWN: 2.0
    }
    
    # Correction guidance templates
    CORRECTION_TEMPLATES = {
        ("shoulder", "flexion"): {
            "raise": ("lower", "Reduce shoulder flexion angle"),
            "lower": ("raise", "Increase shoulder flexion angle")
        },
        ("shoulder", "abduction"): {
            "raise": ("lower", "Reduce shoulder abduction"),
            "lower": ("raise", "Increase shoulder abduction")
        },
        ("elbow", "flexion"): {
            "raise": ("straighten", "Straighten elbow slightly"),
            "lower": ("bend", "Bend elbow more")
        },
        ("elbow", "extension"): {
            "raise": ("lower", "Allow slight bend in elbow"),
            "lower": ("straighten", "Straighten elbow")
        },
        ("wrist", "flexion"): {
            "raise": ("lower", "Reduce wrist flexion"),
            "lower": ("raise", "Increase wrist flexion")
        },
        ("wrist", "extension"): {
            "raise": ("lower", "Reduce wrist extension"),
            "lower": ("raise", "Increase wrist extension")
        }
    }
    
    def __init__(
        self,
        cooldown_enabled: bool = True,
        deduplication_enabled: bool = True,
        max_violation_history: int = 10
    ):
        """
        Initialize the neuro-safe signal engine.
        
        Args:
            cooldown_enabled: Enable cooldown system
            deduplication_enabled: Enable signal deduplication
            max_violation_history: Maximum violation history to track
        """
        self.cooldown_enabled = cooldown_enabled
        self.deduplication_enabled = deduplication_enabled
        
        # Signal history for deduplication
        self._signal_history: deque = deque(maxlen=max_violation_history)
        self._last_signal_code: Optional[str] = None
        self._last_signal_time: float = 0
        self._last_safety_level: Optional[SafetyLevel] = None
        
        # Cooldown tracking
        self._cooldown_until: Dict[SafetyLevel, float] = {
            level: 0 for level in SafetyLevel
        }
        
        # Phase detection
        self._current_phase: ExercisePhase = ExercisePhase.UNKNOWN
        self._phase_start_time: float = 0
        self._movement_detected: bool = False
        
        # Statistics
        self._stats = SignalStatistics()
        
        # Safety threshold for confidence
        self._min_confidence = 0.6
    
    def process_frame(
        self,
        assessment: SafetyAssessment,
        angles: Optional[Dict[str, float]] = None
    ) -> SafetyFrameSignal:
        """
        Process a single frame and emit safety signal.
        
        Args:
            assessment: SafetyAssessment object
            angles: Optional dictionary of joint angles
            
        Returns:
            SafetyFrameSignal with all required fields
        """
        self._stats.total_frames += 1
        timestamp = assessment.timestamp
        frame = assessment.frame_number
        
        # Detect exercise phase
        phase = self._detect_phase(assessment, angles)
        if phase != self._current_phase:
            self._stats.phase_changes += 1
            self._current_phase = phase
            self._phase_start_time = timestamp
        
        # Generate signal components
        safety_flag = assessment.overall_safety.value
        confidence = assessment.confidence
        severity = self.SEVERITY_MAP[assessment.overall_safety].value
        
        # Generate correction guidance
        correction = self._generate_correction(assessment.violations)
        
        # Check deduplication
        signal_code = self._generate_signal_code(assessment)
        is_new, reason = self._should_emit(signal_code, assessment.overall_safety, timestamp)
        
        # Update statistics for all signals (both emitted and suppressed)
        self._update_statistics(assessment.overall_safety)
        
        if not is_new:
            self._stats.suppressed_signals += 1
            return SafetyFrameSignal(
                frame_number=frame,
                timestamp=timestamp,
                safety_flag=safety_flag,
                confidence=confidence,
                severity=severity,
                phase=phase.value,
                correction=None,  # No correction if suppressed
                is_new=False,
                signal_code=signal_code,
                active_violations=len(assessment.violations),
                primary_violation=None
            )
        
        # Update deduplication state
        self._last_signal_code = signal_code
        self._last_signal_time = timestamp
        self._last_safety_level = assessment.overall_safety
        
        # Determine primary violation
        primary_violation = None
        if assessment.violations:
            primary_violation = f"{assessment.violations[0].joint} {assessment.violations[0].movement}"
        
        return SafetyFrameSignal(
            frame_number=frame,
            timestamp=timestamp,
            safety_flag=safety_flag,
            confidence=confidence,
            severity=severity,
            phase=phase.value,
            correction=correction.to_dict() if correction else None,
            is_new=True,
            signal_code=signal_code,
            active_violations=len(assessment.violations),
            primary_violation=primary_violation
        )
    
    def _detect_phase(
        self,
        assessment: SafetyAssessment,
        angles: Optional[Dict[str, float]] = None
    ) -> ExercisePhase:
        """Detect current exercise phase"""
        # Detect movement regardless of safety level
        if angles:
            movement_threshold = 5.0  # degrees of movement
            is_moving = any(
                v > movement_threshold 
                for k, v in angles.items() 
                if 'flexion' in k or 'abduction' in k
            )
            
            if is_moving:
                self._movement_detected = True
                
                # Determine phase based on angle ranges
                flexion_keys = [k for k in angles if 'shoulder_flexion' in k]
                if flexion_keys:
                    avg_flexion = sum(angles[k] for k in flexion_keys) / len(flexion_keys)
                    
                    if avg_flexion < 30:
                        return ExercisePhase.INITIATION
                    elif avg_flexion < 90:
                        return ExercisePhase.ACTIVE
                    else:
                        return ExercisePhase.TRANSITION
        
        # Check if in rest position (arms down, safe)
        if assessment.overall_safety == SafetyLevel.SAFE:
            if not self._movement_detected:
                return ExercisePhase.REST
            
            # Check if returning to rest
            if assessment.confidence > 0.9:
                return ExercisePhase.COMPLETION
        
        return ExercisePhase.ACTIVE if self._movement_detected else ExercisePhase.REST
    
    def _generate_correction(self, violations: List) -> Optional[CorrectionGuidance]:
        """Generate correction guidance for violations"""
        if not violations:
            return None
        
        violation = violations[0]  # Primary violation
        
        joint_parts = violation.joint.split('_')
        joint = joint_parts[-1]  # e.g., "shoulder", "elbow"
        side = joint_parts[0] if len(joint_parts) > 1 else ""
        movement = violation.movement
        
        # Determine correction direction
        current = violation.current_angle
        safe_max = violation.safe_limit
        
        if current > safe_max:
            direction_key = "raise"
            target = safe_max
        else:
            direction_key = "lower"
            target = safe_max
        
        # Get template
        template_key = (joint, movement)
        template = self.CORRECTION_TEMPLATES.get(template_key, {})
        direction_info = template.get(direction_key, ("adjust", f"Adjust {joint} {movement}"))
        
        # Build instruction
        instruction = f"{direction_info[1]}. Current: {current:.1f}°. Target: {target:.1f}°."
        if side:
            instruction = f"{side.upper()} {instruction}"
        
        return CorrectionGuidance(
            joint=f"{side}_{joint}".strip('_'),
            movement=movement,
            direction=direction_info[0],
            target_angle=target,
            instruction=instruction
        )
    
    def _generate_signal_code(self, assessment: SafetyAssessment) -> str:
        """Generate unique signal code for deduplication"""
        level = assessment.overall_safety.value
        
        if assessment.violations:
            # Include violation details in code
            v = assessment.violations[0]
            code = f"{level}_{v.joint}_{v.movement}"
        else:
            code = f"{level}_clean"
        
        # Include confidence level band
        if assessment.confidence >= 0.8:
            code += "_high_conf"
        elif assessment.confidence >= 0.6:
            code += "_med_conf"
        else:
            code += "_low_conf"
        
        # Include phase
        code += f"_{self._current_phase.value}"
        
        return code
    
    def _should_emit(
        self,
        signal_code: str,
        level: SafetyLevel,
        timestamp: float
    ) -> Tuple[bool, str]:
        """
        Determine if signal should be emitted or suppressed.
        
        Returns:
            Tuple of (should_emit, reason)
        """
        # Always emit danger signals
        if level == SafetyLevel.DANGER:
            return True, "danger_signal"
        
        # Check deduplication
        if self.deduplication_enabled:
            # Same signal code as last frame
            if signal_code == self._last_signal_code:
                return False, "duplicate_signal"
        
        # Check cooldown
        if self.cooldown_enabled:
            cooldown = self.COOLDOWN_MAP[level]
            if timestamp < self._cooldown_until.get(level, 0):
                return False, "cooldown"
            
            # Set cooldown
            self._cooldown_until[level] = timestamp + cooldown
        
        return True, "emit"
    
    def _update_statistics(self, level: SafetyLevel):
        """Update signal statistics"""
        if level == SafetyLevel.SAFE:
            self._stats.safe_signals += 1
        elif level == SafetyLevel.WARNING:
            self._stats.warning_signals += 1
        elif level == SafetyLevel.DANGER:
            self._stats.danger_signals += 1
    
    def get_statistics(self) -> Dict:
        """Get signal generation statistics"""
        return self._stats.to_dict()
    
    def reset(self):
        """Reset engine state"""
        self._signal_history.clear()
        self._last_signal_code = None
        self._last_signal_time = 0
        self._last_safety_level = None
        self._current_phase = ExercisePhase.UNKNOWN
        self._movement_detected = False
        self._cooldown_until = {level: 0 for level in SafetyLevel}
        self._stats = SignalStatistics()
    
    def get_output_formats(self, signal: SafetyFrameSignal) -> Dict[str, Any]:
        """
        Get signal in multiple output formats.
        
        Returns:
            Dictionary with json, unreal, vr, minimal formats
        """
        return {
            "json": signal.to_dict(),
            "unreal": self._format_unreal(signal),
            "vr": self._format_vr(signal),
            "minimal": self._format_minimal(signal)
        }
    
    def _format_unreal(self, signal: SafetyFrameSignal) -> Dict:
        """Format for Unreal Engine"""
        return {
            "safety_flag": signal.safety_flag,
            "confidence": signal.confidence,
            "severity": signal.severity,
            "action_required": signal.severity >= 2,
            "command_code": self._get_unreal_command(signal),
            "phase": signal.phase
        }
    
    def _format_vr(self, signal: SafetyFrameSignal) -> Dict:
        """Format for VR systems"""
        return {
            "status": signal.safety_flag,
            "severity": signal.severity,
            "correction": signal.correction,
            "timestamp": signal.timestamp
        }
    
    def _format_minimal(self, signal: SafetyFrameSignal) -> Dict:
        """Minimal format"""
        return {
            "s": signal.safety_flag[0].upper(),  # S, W, D, U
            "v": signal.severity,
            "p": signal.phase[0].upper()  # R, I, A, T, C, U
        }
    
    def _get_unreal_command(self, signal: SafetyFrameSignal) -> str:
        """Get command code for Unreal Engine"""
        if signal.safety_flag == "danger":
            return "STOP_NOW"
        elif signal.safety_flag == "warning":
            return "CORRECT_POSITION"
        else:
            return "CONTINUE"
