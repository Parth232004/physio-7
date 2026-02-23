#!/usr/bin/env python3
"""
PhysioSafe VR Safety System - Demo Script (Demo Flow Hardcoded)

10-15 minute continuous run with:
- Timestamped logging of signals + safety events
- Webcam + console recording
- Session export for analysis
- Hardcoded demo flow for predictable presentation

Demo Flow:
- Minute 1 — Good form (safe range)
- Minute 2 — Slight over-flexion → CAUTION
- Minute 3 — Safe correction
- Minute 4 — Force STOP scenario
- Minute 5 — Recovery
- Minute 6 — Session summary

Usage:
    python run_demo.py --duration 600          # 10 minutes
    python run_demo.py --mock --duration 60    # 1 minute mock test
    python run_demo.py --webcam --duration 300  # 5 minutes with webcam
"""

import sys
import time
import json
import argparse
import os
from datetime import datetime
from typing import Optional

# Import PhysioSafe modules
from main import PhysioSafeSystem
from session_logger import SessionLogger, LogLevel, create_demo_logger
from signal_generator import NeuroSafeSignalEngine


class DemoRunner:
    """
    Demo runner for PhysioSafe VR Safety System.
    
    Features:
    - Configurable duration
    - Session logging
    - Progress reporting
    - Session export
    - Hardcoded demo flow for predictable presentation
    """
    
    # Demo flow phases
    PHASE_GOOD_FORM = "good_form"           # Minute 1
    PHASE_CAUTION = "caution"               # Minute 2
    PHASE_CORRECTION = "correction"         # Minute 3
    PHASE_STOP = "stop"                     # Minute 4
    PHASE_RECOVERY = "recovery"             # Minute 5
    PHASE_SUMMARY = "summary"               # Minute 6
    
    def __init__(
        self,
        duration_seconds: int = 600,  # 10 minutes default
        use_mock: bool = False,
        camera_index: int = 0,
        output_format: str = "json",
        session_id: Optional[str] = None,
        log_dir: str = "logs",
        demo_flow: bool = True
    ):
        """
        Initialize demo runner.
        
        Args:
            duration_seconds: Demo duration in seconds
            use_mock: Use mock tracker
            camera_index: Webcam index
            output_format: Output format
            session_id: Session identifier
            log_dir: Log directory
            demo_flow: Enable hardcoded demo flow
        """
        self.duration_seconds = duration_seconds
        self.use_mock = use_mock
        self.camera_index = camera_index
        self.output_format = output_format
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = log_dir
        self.demo_flow = demo_flow
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = SessionLogger(
            log_file=f"{log_dir}/{self.session_id}_signals.jsonl",
            safety_events_file=f"{log_dir}/{self.session_id}_events.json",
            console_output=True,
            min_log_level=LogLevel.INFO
        )
        
        # Initialize system with demo mode
        self.system = PhysioSafeSystem(
            use_mock_tracker=use_mock,
            camera_index=camera_index,
            output_format=output_format,
            verbose=False,
            cooldown_enabled=True,
            deduplication_enabled=True,
            demo_mode=True,
            config_path="config.json"
        )
        
        # State
        self.start_time = 0
        self.running = False
        self.current_phase = self.PHASE_GOOD_FORM
        self.phase_start_time = 0
        
        # Demo flow narration cues
        self.narration_cues = {
            self.PHASE_GOOD_FORM: "Starting with proper form - arm at safe angle",
            self.PHASE_CAUTION: "Approaching limit - slight over-flexion detected",
            self.PHASE_CORRECTION: "Correcting position - returning to safe range",
            self.PHASE_STOP: "STOP - Dangerous movement detected!",
            self.PHASE_RECOVERY: "Recovery - returning to safe position",
            self.PHASE_SUMMARY: "Session complete - reviewing results"
        }
    
    def _get_current_demo_phase(self, elapsed: float) -> str:
        """Determine current demo phase based on elapsed time"""
        minute = elapsed / 60.0
        
        if minute < 1:
            return self.PHASE_GOOD_FORM
        elif minute < 2:
            return self.PHASE_CAUTION
        elif minute < 3:
            return self.PHASE_CORRECTION
        elif minute < 4:
            return self.PHASE_STOP
        elif minute < 5:
            return self.PHASE_RECOVERY
        else:
            return self.PHASE_SUMMARY
    
    def _print_narration_cue(self, phase: str) -> None:
        """Print narration cue for current phase"""
        if phase in self.narration_cues:
            cue = self.narration_cues[phase]
            print(f"\n{'='*60}")
            print(f"🎤 NARRATION: {cue}")
            print(f"{'='*60}\n")
    
    def run(self):
        """Run the demo"""
        print("=" * 70)
        print("PhysioSafe VR Safety System - Demo Mode (Hardcoded Flow)")
        print("=" * 70)
        print(f"Session ID: {self.session_id}")
        print(f"Duration: {self.duration_seconds} seconds ({self.duration_seconds/60:.1f} minutes)")
        print(f"Mode: {'Mock Tracking' if self.use_mock else 'Webcam Tracking'}")
        print(f"Demo Flow: {'Enabled' if self.demo_flow else 'Disabled'}")
        print(f"Output: {self.output_format}")
        print(f"Log Directory: {self.log_dir}")
        
        if self.demo_flow:
            print("\n📋 Demo Flow:")
            print("  Minute 1 — Good form (safe range)")
            print("  Minute 2 — Slight over-flexion → CAUTION")
            print("  Minute 3 — Safe correction")
            print("  Minute 4 — Force STOP scenario")
            print("  Minute 5 — Recovery")
            print("  Minute 6+ — Session summary")
        
        print("=" * 70)
        
        # Initialize system
        if not self.system.initialize():
            self.logger.log(
                message="Failed to initialize system",
                category="error",
                level=LogLevel.ERROR
            )
            return False
        
        self.logger.log(
            message=f"Demo started - Duration: {self.duration_seconds}s",
            category="demo",
            level=LogLevel.INFO
        )
        
        self.running = True
        self.start_time = time.time()
        self.phase_start_time = self.start_time
        
        # Print initial narration
        self._print_narration_cue(self.current_phase)
        
        try:
            # Run for specified duration
            while self.running:
                elapsed = time.time() - self.start_time
                
                # Check duration
                if elapsed >= self.duration_seconds:
                    self.logger.log(
                        message=f"Demo duration reached ({self.duration_seconds}s)",
                        category="demo",
                        level=LogLevel.INFO
                    )
                    break
                
                # Check for phase change
                new_phase = self._get_current_demo_phase(elapsed)
                if new_phase != self.current_phase:
                    old_phase = self.current_phase
                    self.current_phase = new_phase
                    self.phase_start_time = elapsed
                    
                    # Log phase change
                    self.logger.log_safety_event(
                        event_type="phase_change",
                        description=f"Demo phase change: {old_phase} -> {new_phase}",
                        severity=0,
                        frame_number=self.system.frame_count
                    )
                    
                    # Print narration cue
                    self._print_narration_cue(new_phase)
                
                # Process frame
                self._process_frame()
                
                # Progress update every 30 seconds
                if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                    self._report_progress(elapsed)
        
        except KeyboardInterrupt:
            self.logger.log(
                message="Demo interrupted by user",
                category="demo",
                level=LogLevel.WARNING
            )
        except Exception as e:
            self.logger.log(
                message=f"Demo error: {str(e)}",
                category="error",
                level=LogLevel.ERROR
            )
            raise
        
        finally:
            self._shutdown()
        
        return True
    
    def _process_frame(self):
        """Process a single frame with logging"""
        frame_start = time.time()
        
        # Get current frame from system
        if len(self.system.signals) > 0:
            last_signal = self.system.signals[-1]
            
            # Log signal
            self.logger.log_signal(
                signal_data=last_signal.to_dict(),
                frame_number=last_signal.frame_number
            )
            
            # Log safety events
            if last_signal.safety_flag == "danger":
                self.logger.log_safety_event(
                    event_type="danger",
                    description=f"Danger signal - {last_signal.primary_violation or 'unknown'}",
                    severity=3,
                    frame_number=last_signal.frame_number,
                    signal_data=last_signal.to_dict()
                )
            elif last_signal.safety_flag == "warning":
                self.logger.log_safety_event(
                    event_type="warning",
                    description=f"Warning signal - {last_signal.primary_violation or 'unknown'}",
                    severity=2,
                    frame_number=last_signal.frame_number,
                    signal_data=last_signal.to_dict()
                )
            
            # Log corrections
            if last_signal.correction and last_signal.is_new:
                self.logger.log_correction(
                    joint=last_signal.correction.get("joint", ""),
                    direction=last_signal.correction.get("direction", ""),
                    target=last_signal.correction.get("target_angle", 0),
                    frame_number=last_signal.frame_number
                )
        
        # Update system (runs one frame)
        self.system._process_frame()
    
    def _report_progress(self, elapsed: float):
        """Report demo progress"""
        frames = self.system.frame_count
        fps = frames / elapsed if elapsed > 0 else 0
        remaining = self.duration_seconds - elapsed
        
        stats = self.logger.get_statistics()
        
        # Get current phase
        phase_info = f"Phase: {self.current_phase.replace('_', ' ').title()}"
        
        print(f"\n{'='*50}")
        print(f"PROGRESS REPORT - {elapsed:.0f}s / {self.duration_seconds}s")
        print(f"{'='*50}")
        print(f"{phase_info}")
        print(f"Frames Processed: {frames}")
        print(f"Current FPS: {fps:.1f}")
        print(f"Time Remaining: {remaining:.0f}s")
        print(f"Log Entries: {stats['total_entries']}")
        print(f"Safety Events: {stats['safety_events']}")
        print(f"{'='*50}\n")
        
        self.logger.log(
            message=f"Progress: {elapsed:.0f}s / {self.duration_seconds}s ({100*elapsed/self.duration_seconds:.1f}%)",
            category="progress",
            level=LogLevel.INFO,
            data={
                "frames": frames,
                "fps": round(fps, 1),
                "remaining_seconds": round(remaining, 0),
                "phase": self.current_phase
            }
        )
    
    def _shutdown(self):
        """Shutdown demo and export session"""
        self.running = False
        self.system._shutdown()
        
        # Get final statistics
        elapsed = time.time() - self.start_time
        stats = self.logger.get_statistics()
        
        # Export session
        session_file = f"{self.log_dir}/{self.session_id}_session.json"
        self.logger.export_session(session_file)
        
        # Print summary
        print("\n" + "=" * 70)
        print("DEMO COMPLETE")
        print("=" * 70)
        print(f"Session ID: {self.session_id}")
        print(f"Total Time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"Frames Processed: {self.system.frame_count}")
        print(f"Average FPS: {self.system.frame_count / elapsed:.1f}" if elapsed > 0 else "N/A")
        print(f"Log Entries: {stats['total_entries']}")
        print(f"Safety Events: {stats['safety_events']}")
        print(f"\nOutput Files:")
        print(f"  - {self.log_dir}/{self.session_id}_signals.jsonl")
        print(f"  - {self.log_dir}/{self.session_id}_events.json")
        print(f"  - {session_file}")
        print("=" * 70)
        
        self.logger.log(
            message=f"Demo complete - {elapsed:.1f}s, {self.system.frame_count} frames",
            category="demo",
            level=LogLevel.INFO,
            data={
                "duration_seconds": round(elapsed, 1),
                "frames": self.system.frame_count,
                "fps": round(self.system.frame_count / elapsed, 1) if elapsed > 0 else 0
            }
        )
        
        self.logger.close()


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PhysioSafe VR Safety System - Demo Runner (Hardcoded Flow)"
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=600,
        help="Demo duration in seconds (default: 600 = 10 minutes)"
    )
    parser.add_argument(
        "--mock", "-m",
        action="store_true",
        help="Use mock tracking instead of webcam"
    )
    parser.add_argument(
        "--webcam", "-w",
        action="store_true",
        help="Use webcam (default: uses webcam if available)"
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
        "--session", "-s",
        type=str,
        default=None,
        help="Session ID (auto-generated if not specified)"
    )
    parser.add_argument(
        "--log-dir", "-l",
        type=str,
        default="logs",
        help="Log directory (default: logs)"
    )
    parser.add_argument(
        "--quick-test", "-q",
        action="store_true",
        help="Quick 60-second test"
    )
    parser.add_argument(
        "--no-demo-flow",
        action="store_true",
        help="Disable hardcoded demo flow"
    )
    
    args = parser.parse_args()
    
    # Handle quick test
    if args.quick_test:
        args.duration = 60
        args.mock = True
    
    # Determine tracking mode
    use_mock = args.mock or not args.webcam
    
    # Create and run demo
    runner = DemoRunner(
        duration_seconds=args.duration,
        use_mock=use_mock,
        camera_index=args.camera,
        output_format=args.format,
        session_id=args.session,
        log_dir=args.log_dir,
        demo_flow=not args.no_demo_flow
    )
    
    success = runner.run()
    
    if success:
        print("\n✅ Demo completed successfully!")
        print("Check the log files for detailed session data.")
    else:
        print("\n❌ Demo failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
