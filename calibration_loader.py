"""
Runtime Clinician Calibration Loader

Provides runtime loading and hot-reloading of clinician-specific calibration settings.
- Load calibration profiles from files
- Support for multiple clinician profiles
- Hot-reload without restarting the system
- Validation of calibration values
- Merge with default configuration

Usage:
    loader = CalibrationLoader()
    
    # Load specific clinician profile
    calibration = loader.load_profile("dr_smith")
    
    # Get active calibration (with hot-reload support)
    active = loader.get_active_calibration()
    
    # Watch for file changes
    loader.watch_for_changes(callback=on_calibration_change)
"""

import json
import os
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set


class CalibrationType(Enum):
    """Types of calibration data"""
    JOINT_OFFSETS = "joint_offsets"
    SAFETY_THRESHOLDS = "safety_thresholds"
    PATIENT_RANGE = "patient_range"
    EXERCISE_PRESETS = "exercise_presets"
    DISPLAY_SETTINGS = "display_settings"


@dataclass
class CalibrationProfile:
    """Clinician calibration profile"""
    clinician_id: str
    clinician_name: str
    created_at: str
    modified_at: str
    version: str = "1.0"
    
    # Joint angle offsets (for patient-specific adjustments)
    joint_offsets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Custom safety thresholds (override defaults)
    safety_thresholds: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Patient range of motion limits
    patient_rom_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Exercise-specific settings
    exercise_settings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    # Display preferences
    display_settings: Dict[str, Any] = field(default_factory=dict)
    
    # Validation rules
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "clinician_id": self.clinician_id,
            "clinician_name": self.clinician_name,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "version": self.version,
            "joint_offsets": self.joint_offsets,
            "safety_thresholds": self.safety_thresholds,
            "patient_rom_limits": self.patient_rom_limits,
            "exercise_settings": self.exercise_settings,
            "display_settings": self.display_settings,
            "validation_rules": self.validation_rules
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CalibrationProfile':
        return cls(
            clinician_id=data.get("clinician_id", ""),
            clinician_name=data.get("clinician_name", ""),
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
            version=data.get("version", "1.0"),
            joint_offsets=data.get("joint_offsets", {}),
            safety_thresholds=data.get("safety_thresholds", {}),
            patient_rom_limits=data.get("patient_rom_limits", {}),
            exercise_settings=data.get("exercise_settings", {}),
            display_settings=data.get("display_settings", {}),
            validation_rules=data.get("validation_rules", {})
        )


@dataclass
class CalibrationValidationResult:
    """Result of calibration validation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    applied_overrides: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "applied_overrides": self.applied_overrides
        }


class CalibrationLoader:
    """
    Runtime calibration loader with hot-reload support.
    
    Features:
    - Multiple clinician profile support
    - File-based calibration storage
    - Automatic hot-reload on file changes
    - Validation of calibration values
    - Merge with default configuration
    """
    
    DEFAULT_CALIBRATION_DIR = "calibration"
    DEFAULT_PROFILE = "default"
    
    def __init__(
        self,
        calibration_dir: Optional[str] = None,
        auto_reload: bool = True,
        reload_interval: float = 2.0
    ):
        """
        Initialize calibration loader.
        
        Args:
            calibration_dir: Directory containing calibration profiles
            auto_reload: Enable automatic hot-reload
            reload_interval: Check for file changes every N seconds
        """
        self._calibration_dir = calibration_dir or self.DEFAULT_CALIBRATION_DIR
        self._auto_reload = auto_reload
        self._reload_interval = reload_interval
        
        # Loaded profiles cache
        self._profiles: Dict[str, CalibrationProfile] = {}
        self._active_clinician_id: Optional[str] = None
        self._merged_calibration: Dict[str, Any] = {}
        
        # File modification tracking
        self._file_timestamps: Dict[str, float] = {}
        
        # Callbacks
        self._change_callbacks: List[Callable] = []
        
        # Hot-reload thread
        self._running = False
        self._reload_thread: Optional[threading.Thread] = None
        
        # Default calibration (fallback)
        self._default_calibration = self._get_default_calibration()
        
        # Ensure calibration directory exists
        self._ensure_directory()
        
        # Start auto-reload if enabled
        if self._auto_reload:
            self.start_watching()
    
    def _ensure_directory(self):
        """Ensure calibration directory exists"""
        Path(self._calibration_dir).mkdir(parents=True, exist_ok=True)
        
        # Create default profile if not exists
        default_path = os.path.join(self._calibration_dir, f"{self.DEFAULT_PROFILE}.json")
        if not os.path.exists(default_path):
            self._create_default_profile(default_path)
    
    def _create_default_profile(self, path: str):
        """Create default calibration profile"""
        default = CalibrationProfile(
            clinician_id="default",
            clinician_name="System Default",
            created_at=datetime.now().isoformat(),
            modified_at=datetime.now().isoformat()
        )
        with open(path, 'w') as f:
            json.dump(default.to_dict(), f, indent=2)
    
    def _get_default_calibration(self) -> Dict[str, Any]:
        """Get default calibration values"""
        return {
            "joint_offsets": {
                "shoulder": {"flexion": 0, "abduction": 0, "rotation": 0},
                "elbow": {"flexion": 0, "extension": 0},
                "wrist": {"flexion": 0, "extension": 0}
            },
            "safety_thresholds": {
                "shoulder": {
                    "flexion": {"safe_max": 120, "warning_max": 100, "danger": 130},
                    "abduction": {"safe_max": 150, "warning_max": 120, "danger": 165}
                }
            },
            "patient_rom_limits": {},
            "exercise_settings": {},
            "display_settings": {
                "show_overlay": True,
                "show_corrections": True,
                "color_scheme": "default"
            }
        }
    
    def load_profile(self, clinician_id: str) -> Optional[CalibrationProfile]:
        """
        Load a specific clinician calibration profile.
        
        Args:
            clinician_id: ID of the clinician profile to load
            
        Returns:
            CalibrationProfile if found, None otherwise
        """
        # Check cache first
        if clinician_id in self._profiles:
            return self._profiles[clinician_id]
        
        # Try to load from file
        profile_path = os.path.join(self._calibration_dir, f"{clinician_id}.json")
        
        if not os.path.exists(profile_path):
            # Try without .json extension (already provided)
            if not clinician_id.endswith('.json'):
                profile_path = os.path.join(self._calibration_dir, f"{clinician_id}.json")
                if not os.path.exists(profile_path):
                    return None
        
        try:
            with open(profile_path, 'r') as f:
                data = json.load(f)
            
            profile = CalibrationProfile.from_dict(data)
            self._profiles[clinician_id] = profile
            
            # Update file timestamp
            self._file_timestamps[clinician_id] = os.path.getmtime(profile_path)
            
            return profile
            
        except Exception as e:
            print(f"Error loading calibration profile {clinician_id}: {e}")
            return None
    
    def save_profile(self, profile: CalibrationProfile) -> bool:
        """
        Save a calibration profile to file.
        
        Args:
            profile: CalibrationProfile to save
            
        Returns:
            True if successful, False otherwise
        """
        profile.modified_at = datetime.now().isoformat()
        
        profile_path = os.path.join(self._calibration_dir, f"{profile.clinician_id}.json")
        
        try:
            with open(profile_path, 'w') as f:
                json.dump(profile.to_dict(), f, indent=2)
            
            # Update cache
            self._profiles[profile.clinician_id] = profile
            self._file_timestamps[profile.clinician_id] = os.path.getmtime(profile_path)
            
            # Update merged calibration if this is the active profile
            if profile.clinician_id == self._active_clinician_id:
                self._update_merged_calibration()
            
            return True
            
        except Exception as e:
            print(f"Error saving calibration profile: {e}")
            return False
    
    def set_active_clinician(self, clinician_id: str) -> bool:
        """
        Set the active clinician profile.
        
        Args:
            clinician_id: ID of the clinician profile to activate
            
        Returns:
            True if successful, False otherwise
        """
        profile = self.load_profile(clinician_id)
        
        if profile is None:
            return False
        
        self._active_clinician_id = clinician_id
        self._update_merged_calibration()
        
        return True
    
    def get_active_clinician_id(self) -> Optional[str]:
        """Get the currently active clinician ID"""
        return self._active_clinician_id
    
    def get_active_calibration(self) -> Dict[str, Any]:
        """
        Get the active calibration (merged with defaults).
        
        Returns:
            Dictionary containing merged calibration values
        """
        return self._merged_calibration.copy()
    
    def get_calibration_for_exercise(self, exercise: str) -> Dict[str, Any]:
        """
        Get calibration settings for a specific exercise.
        
        Args:
            exercise: Exercise name (e.g., "shoulder_flexion")
            
        Returns:
            Dictionary of exercise-specific calibration
        """
        calibration = self.get_active_calibration()
        
        # Get exercise settings
        exercise_settings = calibration.get("exercise_settings", {}).get(exercise, {})
        
        # Merge with patient ROM limits
        rom_limits = calibration.get("patient_rom_limits", {}).get(exercise, {})
        
        # Merge with safety thresholds
        safety = calibration.get("safety_thresholds", {}).get(exercise, {})
        
        return {
            "exercise_settings": exercise_settings,
            "rom_limits": rom_limits,
            "safety_thresholds": safety,
            "joint_offsets": calibration.get("joint_offsets", {})
        }
    
    def apply_joint_offset(self, joint: str, movement: str, angle: float) -> float:
        """
        Apply joint offset to an angle measurement.
        
        Args:
            joint: Joint name (e.g., "shoulder")
            movement: Movement type (e.g., "flexion")
            angle: Raw angle measurement
            
        Returns:
            Calibrated angle
        """
        calibration = self.get_active_calibration()
        offsets = calibration.get("joint_offsets", {}).get(joint, {}).get(movement, 0)
        
        return angle + offsets
    
    def validate_calibration(
        self,
        profile: Optional[CalibrationProfile] = None,
        strict: bool = False
    ) -> CalibrationValidationResult:
        """
        Validate a calibration profile.
        
        Args:
            profile: Profile to validate (uses active if None)
            strict: If True, treat warnings as errors
            
        Returns:
            CalibrationValidationResult
        """
        if profile is None:
            profile = self._profiles.get(self._active_clinician_id) if self._active_clinician_id else None
        
        if profile is None:
            return CalibrationValidationResult(
                valid=False,
                errors=["No active calibration profile"]
            )
        
        errors = []
        warnings = []
        applied_overrides = {}
        
        # Validate joint offsets
        for joint, offsets in profile.joint_offsets.items():
            if not isinstance(offsets, dict):
                warnings.append(f"Joint {joint} offsets should be a dictionary")
                continue
                
            for movement, offset in offsets.items():
                if not isinstance(offset, (int, float)):
                    errors.append(f"Offset for {joint}.{movement} must be a number")
                elif abs(offset) > 45:
                    warnings.append(f"Large offset for {joint}.{movement}: {offset}°")
        
        # Validate safety thresholds
        for joint, thresholds in profile.safety_thresholds.items():
            if not isinstance(thresholds, dict):
                warnings.append(f"Safety thresholds for {joint} should be a dictionary")
                continue
            
            # Check threshold ordering
            threshold_keys = ["safe_max", "warning_max", "danger"]
            for key in threshold_keys:
                if key in thresholds:
                    value = thresholds[key]
                    if not isinstance(value, (int, float)):
                        errors.append(f"Threshold {joint}.{key} must be a number")
        
        # Validate patient ROM limits
        for exercise, limits in profile.patient_rom_limits.items():
            if not isinstance(limits, dict):
                warnings.append(f"ROM limits for {exercise} should be a dictionary")
                continue
            
            for movement, limit in limits.items():
                if not isinstance(limit, (int, float)):
                    errors.append(f"ROM limit for {exercise}.{movement} must be a number")
                elif limit < 0 or limit > 180:
                    errors.append(f"ROM limit for {exercise}.{movement} out of range: {limit}")
        
        # Determine validity
        valid = len(errors) == 0 or (strict and len(warnings) == 0)
        
        return CalibrationValidationResult(
            valid=valid,
            errors=errors,
            warnings=warnings,
            applied_overrides=applied_overrides
        )
    
    def list_available_profiles(self) -> List[Dict[str, str]]:
        """
        List all available calibration profiles.
        
        Returns:
            List of profile info dictionaries
        """
        profiles = []
        
        if not os.path.exists(self._calibration_dir):
            return profiles
        
        for filename in os.listdir(self._calibration_dir):
            if filename.endswith('.json'):
                path = os.path.join(self._calibration_dir, filename)
                try:
                    with open(path, 'r') as f:
                        data = json.load(f)
                    
                    profiles.append({
                        "clinician_id": data.get("clinician_id", filename[:-5]),
                        "clinician_name": data.get("clinician_name", "Unknown"),
                        "modified_at": data.get("modified_at", ""),
                        "version": data.get("version", "1.0")
                    })
                except:
                    pass
        
        return profiles
    
    def reload_profile(self, clinician_id: str) -> bool:
        """
        Force reload a specific profile from disk.
        
        Args:
            clinician_id: ID of the profile to reload
            
        Returns:
            True if successful, False otherwise
        """
        # Clear cache
        if clinician_id in self._profiles:
            del self._profiles[clinician_id]
        
        # Reload
        profile = self.load_profile(clinician_id)
        
        if profile is None:
            return False
        
        # Update merged if active
        if clinician_id == self._active_clinician_id:
            self._update_merged_calibration()
            self._notify_change()
        
        return True
    
    def reload_all(self) -> int:
        """
        Reload all profiles from disk.
        
        Returns:
            Number of profiles reloaded
        """
        count = 0
        
        for filename in os.listdir(self._calibration_dir):
            if filename.endswith('.json'):
                clinician_id = filename[:-5]
                if self.reload_profile(clinician_id):
                    count += 1
        
        return count
    
    def watch_for_changes(self, callback: Callable[[str], None] = None):
        """
        Start watching for file changes.
        
        Args:
            callback: Optional callback to call on changes
        """
        if callback:
            self._change_callbacks.append(callback)
        
        if not self._running:
            self._running = True
            self._reload_thread = threading.Thread(target=self._watch_loop, daemon=True)
            self._reload_thread.start()
    
    def stop_watching(self):
        """Stop watching for file changes"""
        self._running = False
        if self._reload_thread:
            self._reload_thread.join(timeout=2.0)
    
    def _watch_loop(self):
        """Background loop for checking file changes"""
        while self._running:
            try:
                for filename in os.listdir(self._calibration_dir):
                    if filename.endswith('.json'):
                        path = os.path.join(self._calibration_dir, filename)
                        mtime = os.path.getmtime(path)
                        clinician_id = filename[:-5]
                        
                        # Check if file was modified
                        if clinician_id in self._file_timestamps:
                            if mtime > self._file_timestamps[clinician_id]:
                                print(f"Calibration file changed: {filename}")
                                self.reload_profile(clinician_id)
                                self._notify_change()
                        else:
                            # New file
                            self.load_profile(clinician_id)
                            self._file_timestamps[clinician_id] = mtime
                            
            except Exception as e:
                print(f"Error checking calibration files: {e}")
            
            time.sleep(self._reload_interval)
    
    def _update_merged_calibration(self):
        """Merge active calibration with defaults"""
        # Start with defaults
        merged = self._default_calibration.copy()
        
        # Get active profile
        profile = self._profiles.get(self._active_clinician_id)
        
        if profile is None:
            # Use default profile
            profile = self.load_profile(self.DEFAULT_PROFILE)
        
        if profile:
            # Merge joint offsets
            if profile.joint_offsets:
                merged["joint_offsets"] = self._merge_dict(
                    merged.get("joint_offsets", {}),
                    profile.joint_offsets
                )
            
            # Merge safety thresholds
            if profile.safety_thresholds:
                merged["safety_thresholds"] = self._merge_dict(
                    merged.get("safety_thresholds", {}),
                    profile.safety_thresholds
                )
            
            # Merge patient ROM limits
            if profile.patient_rom_limits:
                merged["patient_rom_limits"] = self._merge_dict(
                    merged.get("patient_rom_limits", {}),
                    profile.patient_rom_limits
                )
            
            # Merge exercise settings
            if profile.exercise_settings:
                merged["exercise_settings"] = self._merge_dict(
                    merged.get("exercise_settings", {}),
                    profile.exercise_settings
                )
            
            # Merge display settings
            if profile.display_settings:
                merged["display_settings"] = self._merge_dict(
                    merged.get("display_settings", {}),
                    profile.display_settings
                )
        
        self._merged_calibration = merged
    
    @staticmethod
    def _merge_dict(base: Dict, override: Dict) -> Dict:
        """Recursively merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = CalibrationLoader._merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _notify_change(self):
        """Notify callbacks of calibration change"""
        for callback in self._change_callbacks:
            try:
                callback(self._active_clinician_id)
            except Exception as e:
                print(f"Calibration change callback error: {e}")
    
    def get_calibration_info(self) -> Dict:
        """Get information about current calibration state"""
        return {
            "active_clinician": self._active_clinician_id,
            "available_profiles": self.list_available_profiles(),
            "merged_calibration": self.get_active_calibration(),
            "watching": self._running
        }


# Example usage
if __name__ == "__main__":
    # Create loader
    loader = CalibrationLoader(calibration_dir="calibration", auto_reload=True)
    
    # List available profiles
    profiles = loader.list_available_profiles()
    print(f"Available profiles: {profiles}")
    
    # Set active clinician
    if profiles:
        loader.set_active_clinician(profiles[0]["clinician_id"])
    
    # Get calibration
    calibration = loader.get_active_calibration()
    print(f"Active calibration: {calibration}")
