#!/usr/bin/env python3
"""
PhysioSafe VR Safety System - Main Application (Demo Hardened)
Real-time safety monitoring for physiotherapy exercises.

DEMO MODE FEATURES:
- Single exercise lock (shoulder_flexion only)
- Visual overlay with clear status indicators
- Safety override with RED banner
- Crash resistance guards
- Deterministic demo flow
"""

import sys
import time
import json
import argparse
from datetime import datetime
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass

from angle_utils import AngleCalculator, Point3D
from safety_rules import SafetyRules, SafetyLevel
from signal_generator import NeuroSafeSignalEngine, SafetyFrameSignal, ExercisePhase
from pose_tracker import PoseTracker, MockPoseTracker, TrackingState


# ============================================================================
# DEMO MODE CONFIGURATION
# ============================================================================

@dataclass
class DemoConfig:
    """Demo mode configuration"""
    enabled: bool = True
    exercise: str = "shoulder_flexion"
    disable_multi_exercise: bool = True
    disable_advanced_analytics: bool = True


@dataclass
class OverlayConfig:
    """Overlay display configuration"""
    show_angle: bool = True
    show_phase: bool = True
    show_safety: bool = True
    show_correction: bool = True
    show_frame_stability: bool = True
    min_cooldown: float = 2.0


class DemoOverlay:
    """
    Visual overlay for demo mode.
    Displays clear, non-spammy status information.
    """
    
    # ANSI color codes
    RESET = "\033[0m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    BG_RED = "\033[41m"
    BG_YELLOW = "\033[43m"
    BG_GREEN = "\033[42m"
    
    def __init__(self, config: OverlayConfig):
        self.config = config
        self.last_message = ""
        self.last_message_time = 0
        self.frame_stability = "STABLE"
        self.frame_stability_count = 0
    
    def display(
        self,
        angle: float,
        phase: str,
        safety_status: str,
        correction: Optional[Dict] = None,
        frame_number: int = 0,
        fps: float = 0
    ) -> None:
        """Display the overlay with current status"""
        current_time = time.time()
        
        # Determine safety color and symbol
        if safety_status == "danger":
            self._display_danger_banner(angle, phase)
            return
        elif safety_status == "warning":
            color = self.YELLOW
            symbol = "⚠"
            status_text = "CAUTION"
        else:
            color = self.GREEN
            symbol = "✓"
            status_text = "SAFE"
        
        # Cooldown check - prevent message spam
        if current_time - self.last_message_time < self.config.min_cooldown:
            return
        
        self.last_message_time = current_time
        
        # Build status line
        status_line = f"{color}{symbol} {status_text}{self.RESET}"
        
        # Build angle line
        angle_line = ""
        if self.config.show_angle:
            angle_line = f" | Angle: {color}{angle:.1f}°{self.RESET}"
        
        # Build phase line
        phase_line = ""
        if self.config.show_phase:
            phase_line = f" | Phase: {phase.upper()}"
        
        # Build correction line (only one at a time)
        correction_line = ""
        if self.config.show_correction and correction:
            direction = correction.get("direction", "adjust")
            target = correction.get("target_angle", 0)
            correction_line = f" | {self.YELLOW}→ {direction.title()} to {target:.0f}°{self.RESET}"
        
        # Build frame stability line
        stability_line = ""
        if self.config.show_frame_stability:
            stability_line = f" | FPS: {fps:.1f}"
        
        # Print complete status line
        print(f"\r{status_line}{angle_line}{phase_line}{correction_line}{stability_line}    ", end="", flush=True)
    
    def _display_danger_banner(self, angle: float, phase: str) -> None:
        """Display the RED danger banner - suppresses all other messages"""
        banner = f"""
{self.BG_RED}{self.BOLD}{'='*60}
  ⚠⚠⚠  STOP - UNSAFE MOVEMENT  ⚠⚠⚠
{'='*60}
  Current Angle: {angle:.1f}°
  Phase: {phase.upper()}
  
  ⚠ STOP ALL MOVEMENT IMMEDIATELY ⚠
{'='*60}
{self.RESET}
"""
        print(banner)
    
    def update_frame_stability(self, is_stable: bool) -> None:
        """Update frame stability indicator"""
        if is_stable:
            self.frame_stability_count += 1
        else:
            self.frame_stability_count = 0
        
        if self.frame_stability_count > 10:
            self.frame_stability = "STABLE"
        elif self.frame_stability_count > 5:
            self.frame_stability = "OK"
        else:
            self.frame_stability = "UNSTABLE"
    
    def clear(self) -> None:
        """Clear the overlay line"""
        print("\r" + " " * 80 + "\r", end="")


class CrashGuard:
    """
    Crash resistance guards for demo stability.
    Handles edge cases gracefully.
    """
    
    def __init__(self):
        self.errors = []
        self.last_error_time = 0
        self.error_cooldown = 5.0  # 5 seconds between error messages
    
    def handle_no_pose(self) -> bool:
        """Handle case when no pose is detected"""
        current_time = time.time()
        if current_time - self.last_error_time < self.error_cooldown:
            return False
        
        self.last_error_time = current_time
        print(f"\n⚠️ Stand in frame - no pose detected")
        return True
    
    def handle_camera_error(self, error_msg: str = "Camera disconnected") -> bool:
        """Handle camera disconnection gracefully"""
        current_time = time.time()
        if current_time - self.last_error_time < self.error_cooldown:
            return False
        
        self.last_error_time = current_time
        print(f"\n❌ {error_msg}. Attempting reconnection...")
        return True
    
    def handle_zero_division(self, context: str = "angle calculation") -> float:
        """Handle zero division in angle calculation"""
        print(f"\n⚠️ Zero division in {context}. Using default value.")
        return 0.0
    
    def handle_empty_frame(self) -> bool:
        """Handle empty frame read"""
        current_time = time.time()
        if current_time - self.last_error_time < self.error_cooldown:
            return False
        
        self.last_error_time = current_time
        print(f"\n⚠️ Empty frame received - waiting for valid data...")
        return True
    
    def handle_json_error(self, error: Exception) -> bool:
        """Handle JSON write failure"""
        print(f"\n❌ JSON error: {str(error)}. Continuing...")
        return True
    
    def get_error_count(self) -> int:
        """Get total error count"""
        return len(self.errors)


class PhysioSafeSystem:
    """
    Main safety system that coordinates all components.
    Provides real-time safety assessment for physiotherapy exercises.
    DEMO MODE: Locked to single exercise (shoulder_flexion)
    """
    
    def __init__(
        self,
        use_mock_tracker: bool = False,
        camera_index: int = 0,
        output_format: str = "json",
        verbose: bool = False,
        cooldown_enabled: bool = True,
        deduplication_enabled: bool = True,
        demo_mode: bool = True,
        config_path: str = "config.json"
    ):
        """
        Initialize the PhysioSafe system.
        
        Args:
            use_mock_tracker: Use mock tracker instead of webcam
            camera_index: Webcam device index
            output_format: Output format (json, unreal, vr, minimal)
            verbose: Enable verbose output
            cooldown_enabled: Enable cooldown system
            deduplication_enabled: Enable signal deduplication
            demo_mode: Enable demo mode (single exercise, hardened)
            config_path: Path to config file
        """
        self.use_mock = use_mock_tracker
        self.output_format = output_format
        self.verbose = verbose
        self.demo_mode = demo_mode
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Demo mode configuration
        if demo_mode and 'demo_mode' in self.config:
            self.demo_config = DemoConfig(
                enabled=self.config['demo_mode'].get('enabled', True),
                exercise=self.config['demo_mode'].get('exercise', 'shoulder_flexion'),
                disable_multi_exercise=self.config['demo_mode'].get('disable_multi_exercise_switching', True),
                disable_advanced_analytics=self.config['demo_mode'].get('disable_advanced_analytics', True)
            )
        else:
            self.demo_config = DemoConfig(enabled=False)
        
        # Overlay configuration
        if 'output' in self.config and 'overlay' in self.config['output']:
            overlay_cfg = self.config['output']['overlay']
            self.overlay_config = OverlayConfig(
                show_angle=overlay_cfg.get('show_current_angle', True),
                show_phase=overlay_cfg.get('show_phase', True),
                show_safety=overlay_cfg.get('show_safety_status', True),
                show_correction=overlay_cfg.get('show_correction_line', True),
                show_frame_stability=overlay_cfg.get('show_frame_stability', True),
                min_cooldown=overlay_cfg.get('min_message_cooldown_seconds', 2.0)
            )
        else:
            self.overlay_config = OverlayConfig()
        
        # Initialize components
        self.safety_rules = SafetyRules(config_path=config_path)
        self.signal_engine = NeuroSafeSignalEngine(
            cooldown_enabled=cooldown_enabled,
            deduplication_enabled=deduplication_enabled
        )
        
        if use_mock_tracker:
            self.tracker = MockPoseTracker()
        else:
            self.tracker = PoseTracker(camera_index=camera_index)
        
        # Demo-specific components
        self.overlay = DemoOverlay(self.overlay_config)
        self.crash_guard = CrashGuard()
        
        # State tracking
        self.is_running = False
        self.frame_count = 0
        self.start_time = 0
        self.assessments = []
        self.signals = []
        
        # Performance tracking
        self.processing_times = []
        
        # Current angle tracking
        self.current_angle = 0.0
    
    def initialize(self) -> bool:
        """Initialize all components"""
        print("=" * 60)
        print("PhysioSafe VR Safety System - Demo Mode (Hardened)")
        print("=" * 60)
        print(f"Version: 2.0.0 (Demo)")
        print(f"Mode: {'Mock Tracking' if self.use_mock else 'Webcam Tracking'}")
        
        if self.demo_mode and self.demo_config.enabled:
            print(f"Demo Mode: ENABLED")
            print(f"  Exercise: {self.demo_config.exercise}")
            print(f"  Multi-exercise switching: DISABLED")
            print(f"  Advanced analytics: {'ENABLED' if not self.demo_config.disable_advanced_analytics else 'DISABLED'}")
        
        print(f"Output Format: {self.output_format}")
        print(f"Cooldown: {'Enabled' if self.signal_engine.cooldown_enabled else 'Disabled'}")
        print(f"Deduplication: {'Enabled' if self.signal_engine.deduplication_enabled else 'Disabled'}")
        print("=" * 60)
        
        # Initialize tracker
        if not self.tracker.initialize():
            print("Error: Failed to initialize pose tracker")
            return False
        
        print("✓ System initialized successfully")
        print()
        
        return True
    
    def run(self, duration_seconds: Optional[float] = None):
        """
        Run the safety monitoring loop.
        
        Args:
            duration_seconds: Duration limit in seconds (0 or None = infinite)
        """
        if not self.tracker.is_ready():
            print("Error: Tracker not ready")
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        # Display session duration info
        if duration_seconds and duration_seconds > 0:
            print(f"Session duration: {duration_seconds:.0f} seconds")
        else:
            print("Session duration: unlimited (press Ctrl+C to stop)")
        
        print("Starting safety monitoring...")
        print("-" * 60)
        
        try:
            while self.is_running:
                # Check duration limit (0 or None means unlimited)
                if duration_seconds and duration_seconds > 0 and (time.time() - self.start_time) > duration_seconds:
                    break
                
                # Process frame with crash guards
                self._process_frame_safe()
                
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        
        finally:
            self._shutdown()
    
    def _process_frame_safe(self):
        """Process frame with crash guards"""
        try:
            self._process_frame()
        except Exception as e:
            # Handle any unexpected errors gracefully
            self.crash_guard.handle_json_error(e)
            time.sleep(0.1)  # Brief pause before retry
    
    def _process_frame(self):
        """Process a single frame"""
        frame_start = time.time()
        
        # Update tracker with error handling
        try:
            success, landmarks, confidence = self.tracker.update()
        except Exception as e:
            self.crash_guard.handle_camera_error(f"Tracker update failed: {str(e)}")
            return
        
        # Handle no pose detected
        if not success or not landmarks:
            self.crash_guard.handle_no_pose()
            return
        
        # Handle empty landmarks
        if len(landmarks) == 0:
            self.crash_guard.handle_empty_frame()
            return
        
        # Extract angles
        try:
            angles = AngleCalculator.extract_angles(landmarks)
        except Exception as e:
            # Handle zero division or other calculation errors
            self.crash_guard.handle_zero_division("angle extraction")
            angles = {}
        
        if not angles:
            return
        
        # Get the primary angle for demo mode (shoulder_flexion)
        self.current_angle = self._get_primary_angle(angles)
        
        # Calculate overall confidence
        overall_confidence = confidence.overall_confidence() if confidence else 0.5
        
        # Perform safety assessment
        assessment = self.safety_rules.assess_safety(
            angles=angles,
            confidence=overall_confidence,
            frame_number=self.frame_count,
            timestamp=time.time() - self.start_time
        )
        
        # Generate neuro-safe signal
        signal = self.signal_engine.process_frame(assessment, angles)
        
        # Output signal
        self._output_signal(signal)
        
        # Update overlay (DEMO MODE)
        if self.demo_mode and self.demo_config.enabled:
            fps = self.frame_count / (time.time() - self.start_time + 0.001)
            self.overlay.display(
                angle=self.current_angle,
                phase=signal.phase,
                safety_status=signal.safety_flag,
                correction=signal.correction,
                frame_number=signal.frame_number,
                fps=fps
            )
        
        # Store for later analysis
        self.assessments.append(assessment)
        self.signals.append(signal)
        
        # Update frame count
        self.frame_count += 1
        
        # Track processing time
        processing_time = time.time() - frame_start
        self.processing_times.append(processing_time)
        
        # Verbose output
        if self.verbose and self.frame_count % 30 == 0:
            self._print_verbose(signal, assessment)
    
    def _get_primary_angle(self, angles: Dict[str, float]) -> float:
        """
        Get the primary angle for demo mode.
        In demo mode, we only track shoulder_flexion.
        """
        if self.demo_mode and self.demo_config.enabled:
            # Lock to shoulder_flexion only
            target = f"left_{self.demo_config.exercise}"
            if target in angles:
                return angles[target]
            target = f"right_{self.demo_config.exercise}"
            if target in angles:
                return angles[target]
        
        # Fallback: return first available angle
        if angles:
            return list(angles.values())[0]
        return 0.0
    
    def _output_signal(self, signal: SafetyFrameSignal):
        """Output signal in the specified format"""
        if self.output_format == "json":
            output = signal.to_json()
            print(output)
        elif self.output_format == "unreal":
            output = self.signal_engine._format_unreal(signal)
            print(json.dumps(output))
        elif self.output_format == "vr":
            output = self.signal_engine._format_vr(signal)
            print(json.dumps(output))
        elif self.output_format == "minimal":
            output = self.signal_engine._format_minimal(signal)
            print(json.dumps(output))
        else:
            # Default to JSON
            print(signal.to_json())
    
    def _print_verbose(self, signal: SafetyFrameSignal, assessment):
        """Print verbose information"""
        status_map = {
            "safe": "✓ SAFE",
            "warning": "⚠ WARNING",
            "danger": "✗ DANGER",
            "unknown": "? UNKNOWN"
        }
        
        status = status_map.get(signal.safety_flag, "?")
        fps = self.frame_count / (time.time() - self.start_time + 0.001)
        
        print(f"\n[Frame {signal.frame_number}] {status}")
        print(f"  Confidence: {signal.confidence:.1%}")
        print(f"  Severity: {signal.severity}/3")
        print(f"  Phase: {signal.phase}")
        print(f"  Violations: {signal.active_violations}")
        print(f"  FPS: {fps:.1f}")
        
        if signal.correction:
            print(f"  Correction: {signal.correction['instruction']}")
    
    def _shutdown(self):
        """Shutdown and cleanup"""
        self.is_running = False
        self.tracker.release()
        
        print("\n" + "-" * 60)
        print("SESSION SUMMARY")
        print("-" * 60)
        print(f"Total Frames: {self.frame_count}")
        print(f"Duration: {time.time() - self.start_time:.1f} seconds")
        
        if self.processing_times:
            avg_time = sum(self.processing_times) / len(self.processing_times)
            avg_fps = 1.0 / avg_time if avg_time > 0 else 0
            print(f"Average FPS: {avg_fps:.1f}")
            print(f"Average Processing Time: {avg_time*1000:.1f}ms")
        
        # Signal statistics
        stats = self.signal_engine.get_statistics()
        print(f"\nSignal Statistics:")
        print(f"  Safe signals: {stats['safe']}")
        print(f"  Warning signals: {stats['warning']}")
        print(f"  Danger signals: {stats['danger']}")
        print(f"  Suppressed signals: {stats['suppressed']}")
        print(f"  Phase changes: {stats['phase_changes']}")
        
        # Crash guard stats
        print(f"\nCrash Guards:")
        print(f"  Total errors handled: {self.crash_guard.get_error_count()}")
        
        print("\n✓ System shutdown complete")
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "is_running": self.is_running,
            "frame_count": self.frame_count,
            "uptime": time.time() - self.start_time if self.is_running else 0,
            "tracker_state": self.tracker.state.value if hasattr(self.tracker.state, 'value') else str(self.tracker.state),
            "last_signal": self.signals[-1].to_dict() if self.signals else None,
            "output_format": self.output_format,
            "demo_mode": self.demo_mode,
            "current_angle": self.current_angle
        }
    
    def export_data(self, filepath: str):
        """Export assessment data to file"""
        data = {
            "session_info": {
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "frame_count": self.frame_count,
                "output_format": self.output_format,
                "demo_mode": self.demo_mode
            },
            "signals": [s.to_dict() for s in self.signals],
            "statistics": self.signal_engine.get_statistics()
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Data exported to {filepath}")
        except Exception as e:
            self.crash_guard.handle_json_error(e)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PhysioSafe VR Safety System - Demo Mode (Hardened)"
    )
    
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Use mock tracking (no webcam required)"
    )
    parser.add_argument(
        "--camera", "-c",
        type=int,
        default=0,
        help="Camera device index (default: 0)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["json", "unreal", "vr", "minimal"],
        default="json",
        help="Output format (default: json)"
    )
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=60.0,
        help="Duration in seconds (default: 60, set to 0 for unlimited)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--no-cooldown",
        action="store_true",
        help="Disable cooldown system"
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable signal deduplication"
    )
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="Disable demo mode"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Config file path (default: config.json)"
    )
    
    args = parser.parse_args()
    
    # Create and run system
    system = PhysioSafeSystem(
        use_mock_tracker=args.mock,
        camera_index=args.camera,
        output_format=args.format,
        verbose=args.verbose,
        cooldown_enabled=not args.no_cooldown,
        deduplication_enabled=not args.no_dedup,
        demo_mode=not args.no_demo,
        config_path=args.config
    )
    
    if system.initialize():
        # Handle duration: 0 means unlimited, otherwise use specified duration
        duration = None if args.duration == 0 else args.duration
        system.run(duration_seconds=duration)


if __name__ == "__main__":
    main()
