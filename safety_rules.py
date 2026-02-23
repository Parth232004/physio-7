"""
Safety rules and thresholds for physiotherapy exercises.
Defines what's safe and dangerous for upper body movements.
"""

import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from angle_utils import Point3D


class SafetyLevel(Enum):
    """Safety assessment levels"""
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    UNKNOWN = "unknown"


class JointType(Enum):
    """Upper body joint types"""
    SHOULDER_LEFT = "shoulder_left"
    SHOULDER_RIGHT = "shoulder_right"
    ELBOW_LEFT = "elbow_left"
    ELBOW_RIGHT = "elbow_right"
    WRIST_LEFT = "wrist_left"
    WRIST_RIGHT = "wrist_right"


class MovementType(Enum):
    """Types of joint movements"""
    FLEXION = "flexion"
    EXTENSION = "extension"
    HYPEREXTENSION = "hyperextension"
    ABDUCTION = "abduction"
    ADDUCTION = "adduction"
    ROTATION = "rotation"
    RADIAL_DEVIATION = "radial_deviation"
    ULNAR_DEVIATION = "ulnar_deviation"


@dataclass
class SafetyThreshold:
    """Safety threshold for a specific movement"""
    movement: MovementType
    joint: JointType
    safe_max: float
    warning_max: float
    danger_max: float
    unit: str = "degrees"
    
    def check_safety(self, angle: float) -> SafetyLevel:
        """Check safety level based on angle"""
        if angle >= self.danger_max:
            return SafetyLevel.DANGER
        elif angle >= self.warning_max:
            return SafetyLevel.WARNING
        else:
            return SafetyLevel.SAFE


@dataclass
class SafetyViolation:
    """Represents a safety violation"""
    joint: str
    movement: str
    current_angle: float
    safe_limit: float
    safety_level: SafetyLevel
    message: str
    timestamp: float


@dataclass
class SafetyAssessment:
    """Complete safety assessment result"""
    overall_safety: SafetyLevel
    is_safe: bool
    confidence: float
    violations: List[SafetyViolation]
    signals: Dict[str, any]
    timestamp: float
    frame_number: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON output"""
        return {
            "overall_safety": self.overall_safety.value,
            "is_safe": self.is_safe,
            "confidence": round(self.confidence, 3),
            "violation_count": len(self.violations),
            "violations": [
                {
                    "joint": v.joint,
                    "movement": v.movement,
                    "current_angle": round(v.current_angle, 1),
                    "safe_limit": round(v.safe_limit, 1),
                    "safety_level": v.safety_level.value,
                    "message": v.message
                }
                for v in self.violations
            ],
            "signals": self.signals,
            "timestamp": self.timestamp,
            "frame_number": self.frame_number
        }


class SafetyRules:
    """
    Safety rules engine for physiotherapy exercises.
    Defines all safety thresholds and checks for violations.
    """
    
    def __init__(self, config_path: str = "config.json"):
        """Load safety thresholds from config"""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            # Use default config if file not found
            config = self._get_default_config()
        
        self.thresholds = self._build_thresholds(config.get('safety_thresholds', {}))
        self.min_confidence = config.get('confidence', {}).get('min_threshold', 0.6)
        self.suppress_repeated = config.get('output', {}).get('suppress_repeated_warnings', True)
    
    def _get_default_config(self) -> Dict:
        """Get default configuration"""
        return {
            "safety_thresholds": {
                "shoulder": {
                    "flexion": {"safe_max": 120, "warning_max": 100, "danger_threshold": 130, "unit": "degrees"}
                }
            },
            "confidence": {"min_threshold": 0.6},
            "output": {"suppress_repeated_warnings": True}
        }
    
    def _build_thresholds(self, threshold_config: Dict) -> Dict[str, SafetyThreshold]:
        """Build threshold objects from config"""
        thresholds = {}
        
        if not threshold_config:
            return thresholds
        
        # Shoulder thresholds
        if 'shoulder' in threshold_config:
            for movement, values in threshold_config['shoulder'].items():
                for side in ['left', 'right']:
                    key = f"shoulder_{side}_{movement}"
                    thresholds[key] = SafetyThreshold(
                        movement=MovementType(movement),
                        joint=JointType(f"shoulder_{side}"),
                        safe_max=values.get('safe_max', 120),
                        warning_max=values.get('warning_max', 100),
                        danger_max=values.get('danger_threshold', values.get('safe_max', 120)),
                        unit=values.get('unit', 'degrees')
                    )
        
        # Elbow thresholds
        if 'elbow' in threshold_config:
            for movement, values in threshold_config['elbow'].items():
                for side in ['left', 'right']:
                    key = f"elbow_{side}_{movement}"
                    thresholds[key] = SafetyThreshold(
                        movement=MovementType(movement),
                        joint=JointType(f"elbow_{side}"),
                        safe_max=values.get('safe_max', 150),
                        warning_max=values.get('warning_max', 120),
                        danger_max=values.get('danger_threshold', values.get('safe_max', 150) * 1.1),
                        unit=values.get('unit', 'degrees')
                    )
        
        # Wrist thresholds
        if 'wrist' in threshold_config:
            for movement, values in threshold_config['wrist'].items():
                for side in ['left', 'right']:
                    key = f"wrist_{side}_{movement}"
                    thresholds[key] = SafetyThreshold(
                        movement=MovementType(movement),
                        joint=JointType(f"wrist_{side}"),
                        safe_max=values.get('safe_max', 80),
                        warning_max=values.get('warning_max', 60),
                        danger_max=values.get('danger_threshold', values.get('safe_max', 80)),
                        unit=values.get('unit', 'degrees')
                    )
        
        return thresholds
    
    def get_threshold(self, joint: str, movement: str) -> Optional[SafetyThreshold]:
        """Get threshold for a specific joint and movement"""
        key = f"{joint}_{movement}"
        return self.thresholds.get(key)
    
    def check_angle(
        self, 
        joint: str, 
        movement: str, 
        angle: float
    ) -> Tuple[SafetyLevel, Optional[str]]:
        """
        Check if an angle is safe.
        
        Returns:
            Tuple of (safety_level, message)
        """
        threshold = self.get_threshold(joint, movement)
        
        if threshold is None:
            return SafetyLevel.UNKNOWN, f"No threshold defined for {joint} {movement}"
        
        level = threshold.check_safety(angle)
        
        if level == SafetyLevel.DANGER:
            message = f"STOP - {joint} {movement} angle of {angle:.1f}° exceeds danger limit of {threshold.danger_max:.1f}°"
        elif level == SafetyLevel.WARNING:
            message = f"Position needs correction - {joint} {movement} at {angle:.1f}° approaching limit"
        else:
            message = f"{joint} {movement} is safe at {angle:.1f}°"
        
        return level, message
    
    def assess_safety(
        self, 
        angles: Dict[str, float],
        confidence: float,
        frame_number: int = 0,
        timestamp: float = 0.0
    ) -> SafetyAssessment:
        """
        Perform complete safety assessment from all joint angles.
        
        Args:
            angles: Dictionary of angle names to values
            confidence: Confidence score (0-1)
            frame_number: Current frame number
            timestamp: Current timestamp
            
        Returns:
            SafetyAssessment object
        """
        violations = []
        highest_level = SafetyLevel.SAFE
        
        # Check each angle against its threshold
        for angle_name, angle_value in angles.items():
            if angle_value is None:
                continue
                
            parts = angle_name.split('_')
            if len(parts) >= 3:
                joint = '_'.join(parts[:-1])
                movement = parts[-1]
                
                level, message = self.check_angle(joint, movement, angle_value)
                
                if level == SafetyLevel.DANGER:
                    violations.append(SafetyViolation(
                        joint=joint,
                        movement=movement,
                        current_angle=angle_value,
                        safe_limit=self.get_threshold(joint, movement).danger_max if self.get_threshold(joint, movement) else 0,
                        safety_level=level,
                        message=message,
                        timestamp=timestamp
                    ))
                    highest_level = SafetyLevel.DANGER
                elif level == SafetyLevel.WARNING:
                    violations.append(SafetyViolation(
                        joint=joint,
                        movement=movement,
                        current_angle=angle_value,
                        safe_limit=self.get_threshold(joint, movement).warning_max if self.get_threshold(joint, movement) else 0,
                        safety_level=level,
                        message=message,
                        timestamp=timestamp
                    ))
                    if highest_level != SafetyLevel.DANGER:
                        highest_level = SafetyLevel.WARNING
        
        # Generate signals
        signals = self._generate_signals(highest_level, confidence, len(violations))
        
        return SafetyAssessment(
            overall_safety=highest_level,
            is_safe=(highest_level == SafetyLevel.SAFE),
            confidence=confidence,
            violations=violations,
            signals=signals,
            timestamp=timestamp,
            frame_number=frame_number
        )
    
    def _generate_signals(
        self, 
        safety_level: SafetyLevel, 
        confidence: float,
        violation_count: int
    ) -> Dict[str, any]:
        """Generate control signals for VR/UE systems"""
        signals = {
            "safety_status": safety_level.value,
            "action_required": False,
            "action_type": "none",
            "urgency_level": 0,
            "confidence": confidence,
            "message_codes": []
        }
        
        if safety_level == SafetyLevel.DANGER:
            signals["action_required"] = True
            signals["action_type"] = "stop_immediately"
            signals["urgency_level"] = 3
            signals["message_codes"] = ["DANGER_STOP", "EMERGENCY_HALT"]
        elif safety_level == SafetyLevel.WARNING:
            signals["action_required"] = True
            signals["action_type"] = "correct_position"
            signals["urgency_level"] = 2
            signals["message_codes"] = ["WARNING_CORRECT", "POSITION_ADJUST"]
        else:
            signals["action_required"] = False
            signals["action_type"] = "continue"
            signals["urgency_level"] = 0
            signals["message_codes"] = ["SAFE_CONTINUE"]
        
        # Add confidence-based signal
        if confidence < self.min_confidence:
            signals["low_confidence_warning"] = True
            signals["message_codes"].append("UNCERTAIN_ASSESSMENT")
        else:
            signals["low_confidence_warning"] = False
        
        return signals
    
    def get_default_rules(self) -> Dict[str, any]:
        """Get default safety rules as documentation"""
        return {
            "shoulder_flexion": {
                "description": "Forward arm raise",
                "safe_max_degrees": 120,
                "warning_degrees": 100,
                "danger_degrees": 130
            },
            "shoulder_abduction": {
                "description": "Side arm raise",
                "safe_max_degrees": 150,
                "warning_degrees": 120,
                "danger_degrees": 165
            },
            "elbow_flexion": {
                "description": "Bending the elbow",
                "safe_max_degrees": 150,
                "warning_degrees": 120,
                "danger_degrees": 160
            },
            "elbow_extension": {
                "description": "Straightening the elbow",
                "safe_max_degrees": 10,
                "warning_degrees": 5,
                "danger_degrees": -5  # Hyperextension
            },
            "wrist_flexion": {
                "description": "Bending palm toward forearm",
                "safe_max_degrees": 80,
                "warning_degrees": 60,
                "danger_degrees": 90
            },
            "wrist_extension": {
                "description": "Bending back of hand toward forearm",
                "safe_max_degrees": 70,
                "warning_degrees": 55,
                "danger_degrees": 80
            }
        }
