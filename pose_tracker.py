"""
Real-time pose tracking using webcam and MediaPipe.
Provides deterministic landmark detection for upper body analysis.

DEMO MODE: Enhanced with crash guards for stability
"""

import time
import sys
import cv2  # type: ignore
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

# MediaPipe imports (optional - will work without if installed)
try:
    import mediapipe as mp  # type: ignore
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Using fallback tracking.")

from angle_utils import Point3D


class TrackingState(Enum):
    """Tracking state enumeration"""
    NOT_INITIALIZED = "not_initialized"
    INITIALIZING = "initializing"
    TRACKING = "tracking"
    LOST = "lost"
    PAUSED = "paused"


@dataclass
class TrackingConfidence:
    """Confidence scores for tracking quality"""
    pose_detection: float
    pose_presence: float
    visibility_scores: Dict[str, float]
    
    def overall_confidence(self) -> float:
        """Calculate overall tracking confidence"""
        if not self.visibility_scores:
            return self.pose_detection
        
        # Weight by average visibility of key landmarks
        vis_values = list(self.visibility_scores.values())
        avg_visibility = sum(vis_values) / len(vis_values) if vis_values else 0
        
        return (self.pose_detection * 0.5 + self.pose_presence * 0.3 + avg_visibility * 0.2)


class PoseTracker:
    """
    Real-time pose tracking using webcam.
    Provides deterministic landmark positions for upper body analysis.
    """
    
    # Key landmarks for upper body analysis
    KEY_LANDMARKS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist',
        'left_index', 'right_index',
        'left_thumb', 'right_thumb'
    ]
    
    def __init__(
        self, 
        camera_index: int = 0,
        video_source: Optional[str] = None,
        target_fps: int = 30,
        display: bool = True
    ):
        """
        Initialize pose tracker.
        
        Args:
            camera_index: Camera device index
            video_source: Optional video file path
            target_fps: Target FPS for processing
            display: Whether to display webcam feed
        """
        self.camera_index = camera_index
        self.video_source = video_source
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps
        self.display = display
        
        self.state = TrackingState.NOT_INITIALIZED
        self.cap = None
        self.mp_pose = None
        self.pose = None
        self.mp_drawing = None
        self.frame_number = 0
        self.last_frame_time = 0
        
        # Landmark storage
        self.current_landmarks: Dict[str, Point3D] = {}
        self.tracking_confidence: Optional[TrackingConfidence] = None
        
        # Statistics
        self.fps = 0
        self.total_frames = 0
        self.dropped_frames = 0
    
    def _open_video_source(self):
        """Robustly open the video source with backend fallbacks."""
        # If a video file is provided, try that directly
        if self.video_source:
            cap = cv2.VideoCapture(self.video_source)
            if cap.isOpened():
                print(f"Using video file: {self.video_source}")
                return cap
            return None
        
        # Determine backend preferences per OS
        backend_order = []
        if sys.platform == 'darwin':
            backend_order = [getattr(cv2, 'CAP_AVFOUNDATION', 0)]
        elif sys.platform.startswith('win'):
            backend_order = [getattr(cv2, 'CAP_DSHOW', 700), getattr(cv2, 'CAP_MSMF', 1400)]
        else:
            backend_order = [getattr(cv2, 'CAP_V4L2', 200), getattr(cv2, 'CAP_ANY', 0)]
        
        # Try preferred index first, then a few common indices
        indices = [self.camera_index] + [i for i in range(3) if i != self.camera_index]
        for idx in indices:
            for backend in backend_order + [getattr(cv2, 'CAP_ANY', 0)]:
                try:
                    cap = cv2.VideoCapture(idx, backend)
                except TypeError:
                    # Older OpenCV may not support (index, backend) signature
                    cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    print(f"Using camera index {idx} with backend {backend}")
                    return cap
        
        return None
    
    def initialize(self) -> bool:
        """Initialize the pose tracking system"""
        self.state = TrackingState.INITIALIZING
        
        if MEDIAPIPE_AVAILABLE:
            # Initialize MediaPipe Pose
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        
        # Open video source (robust probing across backends and indices)
        self.cap = self._open_video_source()
        
        if not self.cap or not self.cap.isOpened():
            print("Error: Could not open video source (camera).")
            if sys.platform == 'darwin':
                print("macOS note: Ensure the terminal/python process has Camera permission in System Settings > Privacy & Security > Camera.")
            print("Tip: Try a different camera index with --camera, close other apps using the camera, or run with --mock to simulate.")
            self.state = TrackingState.NOT_INITIALIZED
            return False
        
        # Set resolution and target FPS (best-effort)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, float(self.target_fps))
        
        self.state = TrackingState.TRACKING
        print(f"Pose tracker initialized. Target FPS: {self.target_fps}")
        return True
    
    def update(self) -> Tuple[bool, Dict[str, Point3D], Optional[TrackingConfidence]]:
        """
        Update pose tracking with current frame.
        
        Returns:
            Tuple of (success, landmarks_dict, confidence)
        """
        if self.state != TrackingState.TRACKING:
            return False, {}, None
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_interval:
            return True, self.current_landmarks, self.tracking_confidence
        
        self.last_frame_time = current_time
        self.frame_number += 1
        
        try:
            # Read frame
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Attempt a one-time reconnect if frame read fails
                if self.cap:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                self.cap = self._open_video_source()
                if not self.cap or not self.cap.isOpened():
                    self.state = TrackingState.LOST
                    return False, {}, None
                # Retry once after reopening
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    self.state = TrackingState.LOST
                    return False, {}, None
            
            # Process frame
            if MEDIAPIPE_AVAILABLE and self.pose:
                landmarks = self._process_with_mediapipe(frame)
            else:
                landmarks = self._process_fallback(frame)
            
            # Display frame if enabled
            if self.display and frame is not None:
                cv2.imshow('PhysioSafe Pose Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.state = TrackingState.PAUSED
            
            if landmarks:
                self.current_landmarks = landmarks
                return True, landmarks, self.tracking_confidence
            else:
                self.state = TrackingState.LOST
                return False, {}, None
                
        except Exception as e:
            print(f"Error in pose tracking: {e}")
            self.state = TrackingState.LOST
            return False, {}, None
    
    def _process_with_mediapipe(self, frame) -> Dict[str, Point3D]:
        """Process frame using MediaPipe"""
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks and self.mp_drawing:
            # Draw pose landmarks on frame
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),  # Green
                    thickness=2,
                    circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 0),
                    thickness=2
                )
            )
        
        if not results.pose_landmarks:
            return {}
        
        landmarks = {}
        visibility_scores = {}
        
        # Extract landmarks
        for idx, landmark in enumerate(results.pose_landmarks.landmark):
            name = self._get_landmark_name(idx)
            if name:
                landmarks[name] = Point3D(
                    x=landmark.x,  # Normalized 0-1
                    y=landmark.y,
                    z=landmark.z
                )
                visibility_scores[name] = landmark.visibility
        
        # Update confidence
        self.tracking_confidence = TrackingConfidence(
            pose_detection=getattr(results, 'pose_detection_score', 0.9),
            pose_presence=getattr(results, 'pose_presence_score', 0.9),
            visibility_scores=visibility_scores
        )
        
        return landmarks
    
    def _process_fallback(self, frame) -> Dict[str, Point3D]:
        """
        Fallback pose estimation when MediaPipe is not available.
        Uses simple color-based tracking for demonstration.
        """
        # This is a placeholder - in production, use proper ML model
        # For now, return empty to trigger fallback mode
        return {}
    
    def _get_landmark_name(self, index: int) -> Optional[str]:
        """Get landmark name from MediaPipe index"""
        landmark_names = {
            0: 'nose',
            1: 'left_eye_inner', 2: 'left_eye', 3: 'left_eye_outer',
            4: 'right_eye_inner', 5: 'right_eye', 6: 'right_eye_outer',
            7: 'left_ear', 8: 'right_ear',
            9: 'mouth_left', 10: 'mouth_right',
            11: 'left_shoulder', 12: 'right_shoulder',
            13: 'left_elbow', 14: 'right_elbow',
            15: 'left_wrist', 16: 'right_wrist',
            17: 'left_pinky', 18: 'right_pinky',
            19: 'left_index', 20: 'right_index',
            21: 'left_thumb', 22: 'right_thumb',
            23: 'left_hip', 24: 'right_hip',
            25: 'left_knee', 26: 'right_knee',
            27: 'left_ankle', 28: 'right_ankle',
            29: 'left_heel', 30: 'right_heel',
            31: 'left_foot_index', 32: 'right_foot_index'
        }
        return landmark_names.get(index)
    
    def get_upper_body_landmarks(self) -> Dict[str, Point3D]:
        """Get only upper body landmarks"""
        upper_body_keys = [
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_index', 'right_index',
            'left_thumb', 'right_thumb'
        ]
        
        return {k: v for k, v in self.current_landmarks.items() if k in upper_body_keys}
    
    def get_statistics(self) -> Dict:
        """Get tracking statistics"""
        return {
            "state": self.state.value,
            "frame_number": self.frame_number,
            "fps": self.fps,
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "confidence": self.tracking_confidence.overall_confidence() if self.tracking_confidence else 0
        }
    
    def pause(self):
        """Pause tracking"""
        self.state = TrackingState.PAUSED
    
    def resume(self):
        """Resume tracking"""
        if self.state == TrackingState.PAUSED:
            self.state = TrackingState.TRACKING
    
    def release(self):
        """Release resources"""
        if self.cap:
            try:
                self.cap.release()
            finally:
                self.cap = None
        if self.pose:
            try:
                self.pose.close()
            except Exception:
                pass
        if self.display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        self.state = TrackingState.NOT_INITIALIZED
    
    def is_ready(self) -> bool:
        """Check if tracker is ready"""
        return self.state == TrackingState.TRACKING


class MockPoseTracker:
    """
    Mock pose tracker for testing without webcam.
    Generates deterministic landmark positions for testing.
    """
    
    def __init__(self):
        self.state = TrackingState.TRACKING
        self.frame_number = 0
        self.phase = 0
        self.phase_speed = 0.02
    
    def initialize(self) -> bool:
        return True
    
    def update(self) -> Tuple[bool, Dict[str, Point3D], Optional[TrackingConfidence]]:
        """Generate mock upper body landmarks"""
        self.frame_number += 1
        self.phase += self.phase_speed
        
        # Simulate arm movement - oscillating between 45-135 degrees
        angle = 90 + 45 * np.sin(self.phase)
        
        landmarks = {
            'left_shoulder': Point3D(0.3, 0.3, 0),
            'right_shoulder': Point3D(0.7, 0.3, 0),
            'left_elbow': Point3D(
                0.3 + 0.2 * np.cos(np.radians(angle)),
                0.3 + 0.2 * np.sin(np.radians(angle)),
                0
            ),
            'right_elbow': Point3D(
                0.7 - 0.2 * np.cos(np.radians(angle)),
                0.3 + 0.2 * np.sin(np.radians(angle)),
                0
            ),
            'left_wrist': Point3D(
                0.3 + 0.35 * np.cos(np.radians(angle)),
                0.3 + 0.35 * np.sin(np.radians(angle)),
                0
            ),
            'right_wrist': Point3D(
                0.7 - 0.35 * np.cos(np.radians(angle)),
                0.3 + 0.35 * np.sin(np.radians(angle)),
                0
            ),
            'left_index': Point3D(
                0.3 + 0.4 * np.cos(np.radians(angle)),
                0.3 + 0.4 * np.sin(np.radians(angle)),
                0
            ),
            'right_index': Point3D(
                0.7 - 0.4 * np.cos(np.radians(angle)),
                0.3 + 0.4 * np.sin(np.radians(angle)),
                0
            ),
            'left_thumb': Point3D(
                0.3 + 0.38 * np.cos(np.radians(angle)),
                0.3 + 0.38 * np.sin(np.radians(angle)),
                0.02
            ),
            'right_thumb': Point3D(
                0.7 - 0.38 * np.cos(np.radians(angle)),
                0.3 + 0.38 * np.sin(np.radians(angle)),
                0.02
            )
        }
        
        confidence = TrackingConfidence(
            pose_detection=0.95,
            pose_presence=0.98,
            visibility_scores={k: 0.9 for k in landmarks.keys()}
        )
        
        return True, landmarks, confidence
    
    def release(self):
        pass
    
    def is_ready(self) -> bool:
        return True
