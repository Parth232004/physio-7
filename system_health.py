"""
System Health Monitoring Layer

Provides comprehensive health monitoring for the PhysioSafe system:
- FPS monitoring and tracking
- Pose loss detection and recovery tracking
- Latency measurement (capture to processing to output)
- System resource monitoring
- Health status aggregation
- Alerts and threshold violations

Usage:
    health_monitor = SystemHealthMonitor()
    
    # Update with current frame metrics
    health_monitor.update_frame(
        frame_number=1,
        processing_time_ms=15.5,
        pose_detected=True,
        pose_confidence=0.95
    )
    
    # Get health status
    status = health_monitor.get_health_status()
    
    # Subscribe to health events
    health_monitor.on_alert(lambda alert: print(alert))
"""

import time
import threading
import psutil
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable, Dict, List, Optional, Any, Deque


class HealthStatus(Enum):
    """Overall health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class HealthAlert:
    """Health alert event"""
    alert_id: str
    severity: AlertSeverity
    message: str
    metric: str
    value: Any
    threshold: Any
    timestamp: float
    resolved: bool = False
    
    def to_dict(self) -> Dict:
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "message": self.message,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "timestamp": self.timestamp,
            "resolved": self.resolved
        }


@dataclass
class FPSMetrics:
    """FPS-related metrics"""
    current_fps: float = 0
    average_fps: float = 0
    min_fps: float = 0
    max_fps: float = 0
    target_fps: float = 30.0
    frame_time_ms: float = 0
    
    def to_dict(self) -> Dict:
        return {
            "current_fps": round(self.current_fps, 1),
            "average_fps": round(self.average_fps, 1),
            "min_fps": round(self.min_fps, 1),
            "max_fps": round(self.max_fps, 1),
            "target_fps": self.target_fps,
            "frame_time_ms": round(self.frame_time_ms, 2)
        }


@dataclass
class PoseMetrics:
    """Pose tracking metrics"""
    pose_detected: bool = True
    confidence: float = 0.0
    consecutive_losses: int = 0
    total_losses: int = 0
    recovery_time_ms: float = 0
    last_loss_timestamp: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return {
            "pose_detected": self.pose_detected,
            "confidence": round(self.confidence, 3),
            "consecutive_losses": self.consecutive_losses,
            "total_losses": self.total_losses,
            "recovery_time_ms": round(self.recovery_time_ms, 2),
            "last_loss_timestamp": self.last_loss_timestamp
        }


@dataclass
class LatencyMetrics:
    """Latency metrics"""
    capture_to_process_ms: float = 0
    process_to_output_ms: float = 0
    total_latency_ms: float = 0
    average_latency_ms: float = 0
    max_latency_ms: float = 0
    min_latency_ms: float = 0
    
    def to_dict(self) -> Dict:
        return {
            "capture_to_process_ms": round(self.capture_to_process_ms, 2),
            "process_to_output_ms": round(self.process_to_output_ms, 2),
            "total_latency_ms": round(self.total_latency_ms, 2),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "max_latency_ms": round(self.max_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2)
        }


@dataclass
class SystemMetrics:
    """System resource metrics"""
    cpu_percent: float = 0
    memory_percent: float = 0
    memory_used_mb: float = 0
    thread_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "cpu_percent": round(self.cpu_percent, 1),
            "memory_percent": round(self.memory_percent, 1),
            "memory_used_mb": round(self.memory_used_mb, 1),
            "thread_count": self.thread_count
        }


@dataclass
class HealthThresholds:
    """Configurable health thresholds"""
    # FPS thresholds
    min_fps: float = 20.0
    target_fps: float = 30.0
    
    # Pose confidence thresholds
    min_pose_confidence: float = 0.5
    max_consecutive_losses: int = 10
    
    # Latency thresholds (ms)
    max_latency_ms: float = 100.0
    warn_latency_ms: float = 50.0
    
    # System resource thresholds
    max_cpu_percent: float = 90.0
    max_memory_percent: float = 85.0


@dataclass
class SystemHealthStatus:
    """Complete system health status"""
    status: HealthStatus
    overall_score: float  # 0-100
    fps: FPSMetrics
    pose: PoseMetrics
    latency: LatencyMetrics
    system: SystemMetrics
    active_alerts: int = 0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "status": self.status.value,
            "overall_score": round(self.overall_score, 1),
            "fps": self.fps.to_dict(),
            "pose": self.pose.to_dict(),
            "latency": self.latency.to_dict(),
            "system": self.system.to_dict(),
            "active_alerts": self.active_alerts,
            "timestamp": self.timestamp
        }


class SystemHealthMonitor:
    """
    Comprehensive system health monitoring.
    
    Features:
    - Real-time FPS tracking
    - Pose loss detection
    - Latency measurement
    - System resource monitoring
    - Configurable thresholds
    - Alert system with callbacks
    """
    
    def __init__(
        self,
        thresholds: Optional[HealthThresholds] = None,
        history_size: int = 300  # 10 seconds at 30fps
    ):
        """
        Initialize health monitor.
        
        Args:
            thresholds: Custom health thresholds
            history_size: Number of frames to keep in history
        """
        self._thresholds = thresholds or HealthThresholds()
        self._history_size = history_size
        
        # Metrics
        self._fps_metrics = FPSMetrics()
        self._pose_metrics = PoseMetrics()
        self._latency_metrics = LatencyMetrics()
        self._system_metrics = SystemMetrics()
        
        # History for averaging
        self._fps_history: Deque[float] = deque(maxlen=history_size)
        self._latency_history: Deque[float] = deque(maxlen=history_size)
        
        # Timing
        self._frame_times: Deque[float] = deque(maxlen=30)
        self._last_frame_time: float = 0
        self._start_time: float = time.time()
        
        # Frame tracking
        self._frame_count: int = 0
        self._last_pose_detect_time: Optional[float] = None
        
        # Processing time tracking
        self._processing_start_time: float = 0
        
        # Alerts
        self._active_alerts: Dict[str, HealthAlert] = {}
        self._alert_callbacks: List[Callable[[HealthAlert], None]] = []
        self._alert_history: Deque[HealthAlert] = deque(maxlen=100)
        
        # Lock for thread safety
        self._lock = threading.Lock()
        
        # System process
        try:
            self._process = psutil.Process(os.getpid())
        except:
            self._process = None
    
    def update_frame(
        self,
        frame_number: int,
        processing_time_ms: float,
        pose_detected: bool,
        pose_confidence: float,
        capture_time: Optional[float] = None,
        output_time: Optional[float] = None
    ):
        """
        Update health metrics with new frame data.
        
        Args:
            frame_number: Current frame number
            processing_time_ms: Time to process this frame
            pose_detected: Whether pose was detected
            pose_confidence: Pose detection confidence (0-1)
            capture_time: Timestamp when frame was captured
            output_time: Timestamp when output was generated
        """
        with self._lock:
            current_time = time.time()
            self._frame_count = frame_number
            
            # Update FPS
            self._update_fps(current_time)
            
            # Update pose metrics
            self._update_pose(pose_detected, pose_confidence, current_time)
            
            # Update latency
            self._update_latency(
                processing_time_ms,
                capture_time or current_time,
                output_time or current_time
            )
            
            # Update system metrics
            self._update_system_metrics()
    
    def _update_fps(self, current_time: float):
        """Update FPS metrics"""
        if self._last_frame_time > 0:
            frame_delta = current_time - self._last_frame_time
            
            if frame_delta > 0:
                current_fps = 1.0 / frame_delta
                self._fps_history.append(current_fps)
                
                self._fps_metrics.current_fps = current_fps
                self._fps_metrics.average_fps = sum(self._fps_history) / len(self._fps_history)
                self._fps_metrics.min_fps = min(self._fps_history) if self._fps_history else 0
                self._fps_metrics.max_fps = max(self._fps_history) if self._fps_history else 0
                self._fps_metrics.frame_time_ms = frame_delta * 1000
        
        self._last_frame_time = current_time
    
    def _update_pose(self, detected: bool, confidence: float, current_time: float):
        """Update pose tracking metrics"""
        was_detected = self._pose_metrics.pose_detected
        
        self._pose_metrics.pose_detected = detected
        self._pose_metrics.confidence = confidence
        
        if detected:
            if not was_detected:
                # Pose recovered
                if self._pose_metrics.last_loss_timestamp:
                    self._pose_metrics.recovery_time_ms = (
                        current_time - self._pose_metrics.last_loss_timestamp
                    ) * 1000
                self._pose_metrics.consecutive_losses = 0
            self._last_pose_detect_time = current_time
        else:
            self._pose_metrics.consecutive_losses += 1
            self._pose_metrics.total_losses += 1
            self._pose_metrics.last_loss_timestamp = current_time
    
    def _update_latency(
        self,
        processing_time_ms: float,
        capture_time: float,
        output_time: float
    ):
        """Update latency metrics"""
        capture_to_process = processing_time_ms
        process_to_output = (output_time - capture_time) * 1000 - processing_time_ms
        total = (output_time - capture_time) * 1000
        
        self._latency_metrics.capture_to_process_ms = capture_to_process
        self._latency_metrics.process_to_output_ms = max(0, process_to_output)
        self._latency_metrics.total_latency_ms = max(0, total)
        
        if total > 0:
            self._latency_history.append(total)
            
            if len(self._latency_history) > 1:
                self._latency_metrics.average_latency_ms = sum(self._latency_history) / len(self._latency_history)
                self._latency_metrics.max_latency_ms = max(self._latency_history)
                self._latency_metrics.min_latency_ms = min(self._latency_history)
    
    def _update_system_metrics(self):
        """Update system resource metrics"""
        try:
            if self._process:
                self._system_metrics.cpu_percent = self._process.cpu_percent()
                mem_info = self._process.memory_info()
                self._system_metrics.memory_used_mb = mem_info.rss / (1024 * 1024)
                self._system_metrics.memory_percent = self._process.memory_percent()
                self._system_metrics.thread_count = self._process.num_threads()
            else:
                # Use psutil for system-wide
                self._system_metrics.cpu_percent = psutil.cpu_percent()
                mem = psutil.virtual_memory()
                self._system_metrics.memory_percent = mem.percent
                self._system_metrics.memory_used_mb = mem.used / (1024 * 1024)
        except:
            pass
    
    def get_health_status(self) -> SystemHealthStatus:
        """
        Get current health status.
        
        Returns:
            SystemHealthStatus with all metrics
        """
        with self._lock:
            # Check thresholds and generate alerts
            self._check_thresholds()
            
            # Calculate overall score
            score = self._calculate_health_score()
            
            # Determine status
            status = self._determine_status(score)
            
            return SystemHealthStatus(
                status=status,
                overall_score=score,
                fps=self._fps_metrics,
                pose=self._pose_metrics,
                latency=self._latency_metrics,
                system=self._system_metrics,
                active_alerts=len(self._active_alerts)
            )
    
    def _calculate_health_score(self) -> float:
        """Calculate overall health score (0-100)"""
        score = 100.0
        
        # FPS penalty (max 30 points)
        if self._fps_metrics.average_fps < self._thresholds.target_fps:
            fps_ratio = max(0, self._fps_metrics.average_fps / self._thresholds.target_fps)
            score -= (1 - fps_ratio) * 30
        
        # Pose confidence penalty (max 25 points)
        if self._pose_metrics.confidence < self._thresholds.min_pose_confidence:
            conf_ratio = max(0, self._pose_metrics.confidence / self._thresholds.min_pose_confidence)
            score -= (1 - conf_ratio) * 25
        
        # Pose loss penalty (max 25 points)
        if self._pose_metrics.consecutive_losses > 0:
            loss_ratio = min(1.0, self._pose_metrics.consecutive_losses / self._thresholds.max_consecutive_losses)
            score -= loss_ratio * 25
        
        # Latency penalty (max 10 points)
        if self._latency_metrics.average_latency_ms > self._thresholds.warn_latency_ms:
            latency_ratio = min(1.0, self._latency_metrics.average_latency_ms / self._thresholds.max_latency_ms)
            score -= latency_ratio * 10
        
        # System resource penalty (max 10 points)
        cpu_ratio = self._system_metrics.cpu_percent / 100.0
        mem_ratio = self._system_metrics.memory_percent / 100.0
        resource_penalty = max(cpu_ratio, mem_ratio) * 10
        score -= resource_penalty
        
        return max(0, min(100, score))
    
    def _determine_status(self, score: float) -> HealthStatus:
        """Determine health status from score"""
        if score >= 80:
            return HealthStatus.HEALTHY
        elif score >= 50:
            return HealthStatus.DEGRADED
        elif score > 0:
            return HealthStatus.CRITICAL
        else:
            return HealthStatus.UNKNOWN
    
    def _check_thresholds(self):
        """Check thresholds and generate alerts"""
        # FPS alerts
        if self._fps_metrics.average_fps < self._thresholds.min_fps:
            self._create_alert(
                "low_fps",
                AlertSeverity.ERROR if self._fps_metrics.average_fps < 15 else AlertSeverity.WARNING,
                "FPS below minimum threshold",
                "fps",
                self._fps_metrics.average_fps,
                self._thresholds.min_fps
            )
        else:
            self._resolve_alert("low_fps")
        
        # Pose confidence alerts
        if self._pose_metrics.confidence < self._thresholds.min_pose_confidence:
            self._create_alert(
                "low_pose_confidence",
                AlertSeverity.WARNING,
                "Pose confidence below threshold",
                "pose_confidence",
                self._pose_metrics.confidence,
                self._thresholds.min_pose_confidence
            )
        else:
            self._resolve_alert("low_pose_confidence")
        
        # Pose loss alerts
        if self._pose_metrics.consecutive_losses >= self._thresholds.max_consecutive_losses:
            self._create_alert(
                "pose_loss",
                AlertSeverity.CRITICAL,
                "Consecutive pose losses exceeded threshold",
                "consecutive_losses",
                self._pose_metrics.consecutive_losses,
                self._thresholds.max_consecutive_losses
            )
        elif self._pose_metrics.consecutive_losses > 0:
            self._resolve_alert("pose_loss")
        
        # Latency alerts
        if self._latency_metrics.average_latency_ms > self._thresholds.max_latency_ms:
            self._create_alert(
                "high_latency",
                AlertSeverity.ERROR,
                "Latency above maximum threshold",
                "latency",
                self._latency_metrics.average_latency_ms,
                self._thresholds.max_latency_ms
            )
        elif self._latency_metrics.average_latency_ms > self._thresholds.warn_latency_ms:
            self._resolve_alert("high_latency")
            self._create_alert(
                "moderate_latency",
                AlertSeverity.WARNING,
                "Latency above warning threshold",
                "latency",
                self._latency_metrics.average_latency_ms,
                self._thresholds.warn_latency_ms
            )
        else:
            self._resolve_alert("high_latency")
            self._resolve_alert("moderate_latency")
        
        # System resource alerts
        if self._system_metrics.cpu_percent > self._thresholds.max_cpu_percent:
            self._create_alert(
                "high_cpu",
                AlertSeverity.WARNING,
                "CPU usage above threshold",
                "cpu_percent",
                self._system_metrics.cpu_percent,
                self._thresholds.max_cpu_percent
            )
        else:
            self._resolve_alert("high_cpu")
        
        if self._system_metrics.memory_percent > self._thresholds.max_memory_percent:
            self._create_alert(
                "high_memory",
                AlertSeverity.WARNING,
                "Memory usage above threshold",
                "memory_percent",
                self._system_metrics.memory_percent,
                self._thresholds.max_memory_percent
            )
        else:
            self._resolve_alert("high_memory")
    
    def _create_alert(
        self,
        alert_id: str,
        severity: AlertSeverity,
        message: str,
        metric: str,
        value: Any,
        threshold: Any
    ):
        """Create a new alert"""
        if alert_id not in self._active_alerts:
            alert = HealthAlert(
                alert_id=alert_id,
                severity=severity,
                message=message,
                metric=metric,
                value=value,
                threshold=threshold,
                timestamp=time.time()
            )
            self._active_alerts[alert_id] = alert
            self._alert_history.append(alert)
            self._notify_alert(alert)
    
    def _resolve_alert(self, alert_id: str):
        """Resolve an existing alert"""
        if alert_id in self._active_alerts:
            alert = self._active_alerts[alert_id]
            alert.resolved = True
            del self._active_alerts[alert_id]
    
    def _notify_alert(self, alert: HealthAlert):
        """Notify callbacks of new alert"""
        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                print(f"Alert callback error: {e}")
    
    def on_alert(self, callback: Callable[[HealthAlert], None]):
        """Register alert callback"""
        self._alert_callbacks.append(callback)
    
    def get_active_alerts(self) -> List[Dict]:
        """Get list of active alerts"""
        return [alert.to_dict() for alert in self._active_alerts.values()]
    
    def get_alert_history(self, limit: int = 50) -> List[Dict]:
        """Get alert history"""
        alerts = list(self._alert_history)[-limit:]
        return [alert.to_dict() for alert in alerts]
    
    def get_summary(self) -> Dict:
        """Get health summary"""
        status = self.get_health_status()
        return {
            "status": status.status.value,
            "score": status.overall_score,
            "frame_count": self._frame_count,
            "uptime_seconds": time.time() - self._start_time,
            "fps": status.fps.to_dict(),
            "pose": status.pose.to_dict(),
            "latency": status.latency.to_dict(),
            "active_alerts": len(self._active_alerts)
        }
    
    def reset(self):
        """Reset all metrics"""
        with self._lock:
            self._fps_metrics = FPSMetrics()
            self._pose_metrics = PoseMetrics()
            self._latency_metrics = LatencyMetrics()
            self._fps_history.clear()
            self._latency_history.clear()
            self._active_alerts.clear()
            self._frame_count = 0
            self._start_time = time.time()


# VR Integration Helper
class VRHealthIntegration:
    """
    Helper class for integrating health monitoring with VR/Unreal streaming.
    """
    
    def __init__(self, health_monitor: SystemHealthMonitor):
        self._health_monitor = health_monitor
    
    def get_vr_friendly_status(self) -> Dict:
        """
        Get health status in VR-friendly format.
        
        Returns:
            Dictionary optimized for VR display
        """
        status = self._health_monitor.get_health_status()
        
        # Map health status to VR indicators
        status_indicator = {
            HealthStatus.HEALTHY: 0,      # Green
            HealthStatus.DEGRADED: 1,     # Yellow
            HealthStatus.CRITICAL: 2,     # Red
            HealthStatus.UNKNOWN: 3       # Gray
        }
        
        return {
            "status_code": status_indicator.get(status.status, 3),
            "health_score": status.overall_score,
            "fps": round(status.fps.current_fps, 0),
            "latency_ms": round(status.latency.total_latency_ms, 0),
            "pose_tracking_ok": status.pose.pose_detected,
            "pose_confidence": status.pose.confidence,
            "has_warnings": len(self._health_monitor.get_active_alerts()) > 0
        }
    
    def check_vr_ready(self) -> Dict:
        """
        Check if system is ready for VR streaming.
        
        Returns:
            Dictionary with VR readiness status
        """
        status = self._health_monitor.get_health_status()
        
        ready = (
            status.fps.current_fps >= 25 and
            status.pose.pose_detected and
            status.latency.total_latency_ms < 100 and
            status.status != HealthStatus.CRITICAL
        )
        
        issues = []
        if status.fps.current_fps < 25:
            issues.append("Low FPS")
        if not status.pose.pose_detected:
            issues.append("Pose not detected")
        if status.latency.total_latency_ms >= 100:
            issues.append("High latency")
        if status.status == HealthStatus.CRITICAL:
            issues.append("System in critical state")
        
        return {
            "vr_ready": ready,
            "issues": issues,
            "health_score": status.overall_score
        }
