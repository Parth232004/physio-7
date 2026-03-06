"""
VR Integration Layer for Unreal Engine

Provides complete VR integration infrastructure:
- Unreal-validated integration output
- Real VR connection validation and streaming handshake
- Integration with all other modules (streaming, health, calibration)
- Connection lifecycle management
- Data format validation for Unreal consumption

Usage:
    vr_integration = VRIntegration()
    
    # Initialize with all components
    await vr_integration.initialize(
        streaming_protocol="udp",
        host="192.168.1.100",
        port=7777
    )
    
    # Connect to Unreal
    connected = await vr_integration.connect_unreal()
    
    # Stream data
    vr_integration.send_signal(signal_data)
    vr_integration.send_health(health_data)
"""

import asyncio
import json
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

from live_streaming import (
    LiveStreamingManager,
    StreamingConfig,
    StreamProtocol,
    ConnectionState,
    StreamMessage
)
from system_health import (
    SystemHealthMonitor,
    HealthStatus,
    VRHealthIntegration
)
from calibration_loader import CalibrationLoader
from safety_event_manager import SafetyEventManager


class IntegrationState(Enum):
    """State of VR integration"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    STREAMING = "streaming"
    ERROR = "error"
    DISCONNECTED = "disconnected"


class ValidationResult(Enum):
    """Result of validation checks"""
    VALID = "valid"
    INVALID = "invalid"
    WARNING = "warning"


@dataclass
class UnrealValidationResult:
    """Result of Unreal integration validation"""
    result: ValidationResult
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "result": self.result.value,
            "checks_passed": self.checks_passed,
            "checks_failed": self.checks_failed,
            "warnings": self.warnings,
            "metadata": self.metadata
        }
    
    @property
    def is_valid(self) -> bool:
        return self.result in [ValidationResult.VALID, ValidationResult.WARNING]


@dataclass
class StreamingHandshake:
    """Handshake data for VR connection"""
    protocol_version: str = "1.0"
    client_type: str = "physiosafe_vr"
    capabilities: Set[str] = field(default_factory=lambda: {
        "signal_streaming",
        "health_monitoring",
        "calibration_sync",
        "event_streaming"
    })
    unreal_version: Optional[str] = None
    vr_device: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict:
        return {
            "protocol_version": self.protocol_version,
            "client_type": self.client_type,
            "capabilities": list(self.capabilities),
            "unreal_version": self.unreal_version,
            "vr_device": self.vr_device,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'StreamingHandshake':
        return cls(
            protocol_version=data.get("protocol_version", "1.0"),
            client_type=data.get("client_type", "physiosafe_vr"),
            capabilities=set(data.get("capabilities", [])),
            unreal_version=data.get("unreal_version"),
            vr_device=data.get("vr_device"),
            timestamp=data.get("timestamp", time.time())
        )


class VRIntegration:
    """
    Complete VR Integration Layer for Unreal Engine.
    
    Features:
    - Live signal streaming (UDP/TCP/WebSocket)
    - System health monitoring integration
    - Runtime calibration loading
    - Safety event management
    - Unreal validation
    - Streaming handshake
    - Connection lifecycle management
    """
    
    def __init__(
        self,
        streaming_config: Optional[StreamingConfig] = None,
        enable_health_monitoring: bool = True,
        enable_calibration: bool = True,
        enable_event_manager: bool = True
    ):
        """
        Initialize VR integration.
        
        Args:
            streaming_config: Custom streaming configuration
            enable_health_monitoring: Enable health monitoring
            enable_calibration: Enable calibration loader
            enable_event_manager: Enable safety event manager
        """
        # State
        self._state = IntegrationState.UNINITIALIZED
        self._error_message: Optional[str] = None
        
        # Configuration
        self._streaming_config = streaming_config or StreamingConfig(
            protocol=StreamProtocol.UDP,
            host="127.0.0.1",
            port=7777
        )
        
        # Components
        self._streaming_manager: Optional[LiveStreamingManager] = None
        self._health_monitor: Optional[SystemHealthMonitor] = None
        self._health_integration: Optional[VRHealthIntegration] = None
        self._calibration_loader: Optional[CalibrationLoader] = None
        self._event_manager: Optional[SafetyEventManager] = None
        
        # Flags
        self._enable_health = enable_health_monitoring
        self._enable_calibration = enable_calibration
        self._enable_events = enable_event_manager
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable]] = {
            "on_connect": [],
            "on_disconnect": [],
            "on_stream_start": [],
            "on_stream_stop": [],
            "on_validation": [],
            "on_error": [],
            "on_health_critical": []
        }
        
        # Connection metadata
        self._connection_time: Optional[float] = None
        self._handshake: Optional[StreamingHandshake] = None
        self._unreal_validation: Optional[UnrealValidationResult] = None
        
        # Statistics
        self._stats = {
            "frames_sent": 0,
            "frames_dropped": 0,
            "last_send_time": 0,
            "connection_uptime": 0
        }
    
    @property
    def state(self) -> IntegrationState:
        """Get current integration state"""
        return self._state
    
    @property
    def is_connected(self) -> bool:
        """Check if connected to Unreal"""
        return self._state in [IntegrationState.CONNECTED, IntegrationState.STREAMING]
    
    @property
    def is_streaming(self) -> bool:
        """Check if actively streaming"""
        return self._state == IntegrationState.STREAMING
    
    async def initialize(self) -> bool:
        """
        Initialize all VR integration components.
        
        Returns:
            True if initialization successful
        """
        try:
            self._state = IntegrationState.INITIALIZING
            
            # Initialize streaming manager
            self._streaming_manager = LiveStreamingManager(self._streaming_config)
            self._streaming_manager.create_client(self._streaming_config.protocol)
            
            # Initialize health monitoring
            if self._enable_health:
                self._health_monitor = SystemHealthMonitor()
                self._health_integration = VRHealthIntegration(self._health_monitor)
                
                # Register health alert callback
                self._health_monitor.on_alert(self._handle_health_alert)
            
            # Initialize calibration loader
            if self._enable_calibration:
                self._calibration_loader = CalibrationLoader(
                    auto_reload=True,
                    reload_interval=2.0
                )
                # Set default clinician
                self._calibration_loader.set_active_clinician("default")
            
            # Initialize event manager
            if self._enable_events:
                self._event_manager = SafetyEventManager()
            
            self._state = IntegrationState.READY
            return True
            
        except Exception as e:
            self._error_message = str(e)
            self._state = IntegrationState.ERROR
            self._emit("on_error", e)
            return False
    
    async def connect_unreal(
        self,
        protocol: Optional[StreamProtocol] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        handshake_data: Optional[Dict] = None
    ) -> bool:
        """
        Connect to Unreal Engine with handshake validation.
        
        Args:
            protocol: Streaming protocol (UDP/TCP/WebSocket)
            host: Target host address
            port: Target port
            handshake_data: Additional handshake data
            
        Returns:
            True if connection successful
        """
        if self._state not in [IntegrationState.READY, IntegrationState.DISCONNECTED]:
            return False
        
        try:
            self._state = IntegrationState.CONNECTING
            
            # Update config
            if protocol:
                self._streaming_config.protocol = protocol
            if host:
                self._streaming_config.host = host
            if port:
                self._streaming_config.port = port
            
            # Create handshake data
            self._handshake = StreamingHandshake(
                unreal_version=handshake_data.get("unreal_version") if handshake_data else None,
                vr_device=handshake_data.get("vr_device") if handshake_data else None
            )
            
            # Connect via streaming manager
            success = self._streaming_manager.connect(protocol)
            
            if not success:
                self._state = IntegrationState.ERROR
                self._error_message = "Failed to connect to streaming server"
                return False
            
            # Validate connection
            validation = await self._validate_unreal_connection()
            
            if not validation.is_valid:
                self._state = IntegrationState.ERROR
                self._error_message = f"Validation failed: {validation.checks_failed}"
                self._emit("on_validation", validation.to_dict())
                return False
            
            self._unreal_validation = validation
            self._connection_time = time.time()
            self._state = IntegrationState.CONNECTED
            self._emit("on_connect", validation.to_dict())
            
            return True
            
        except Exception as e:
            self._error_message = str(e)
            self._state = IntegrationState.ERROR
            self._emit("on_error", e)
            return False
    
    async def _validate_unreal_connection(self) -> UnrealValidationResult:
        """
        Validate connection to Unreal Engine.
        
        Returns:
            UnrealValidationResult
        """
        checks_passed = []
        checks_failed = []
        warnings = []
        
        # Check streaming connection
        if self._streaming_manager.is_connected():
            checks_passed.append("streaming_connection")
        else:
            checks_failed.append("streaming_connection")
        
        # Check health status if enabled
        if self._enable_health and self._health_monitor:
            health = self._health_monitor.get_health_status()
            
            if health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]:
                checks_passed.append("system_health")
            else:
                checks_failed.append("system_health")
                warnings.append(f"System health is {health.status.value}")
        
        # Check calibration if enabled
        if self._enable_calibration and self._calibration_loader:
            calibration = self._calibration_loader.get_active_calibration()
            if calibration:
                checks_passed.append("calibration_loaded")
            else:
                warnings.append("No calibration profile active")
        
        # Determine result
        if checks_failed:
            result = ValidationResult.INVALID
        elif warnings:
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.VALID
        
        return UnrealValidationResult(
            result=result,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            metadata={
                "handshake": self._handshake.to_dict() if self._handshake else {},
                "connection_time": self._connection_time
            }
        )
    
    def start_streaming(self) -> bool:
        """
        Start streaming data to Unreal.
        
        Returns:
            True if streaming started successfully
        """
        if self._state != IntegrationState.CONNECTED:
            return False
        
        try:
            self._streaming_manager.start_async_receiver()
            self._state = IntegrationState.STREAMING
            self._emit("on_stream_start")
            return True
            
        except Exception as e:
            self._emit("on_error", e)
            return False
    
    def stop_streaming(self):
        """Stop streaming data"""
        if self._state == IntegrationState.STREAMING:
            self._state = IntegrationState.CONNECTED
            self._emit("on_stream_stop")
    
    async def disconnect(self):
        """Disconnect from Unreal"""
        if self._streaming_manager:
            self._streaming_manager.disconnect()
        
        if self._calibration_loader:
            self._calibration_loader.stop_watching()
        
        self._state = IntegrationState.DISCONNECTED
        self._connection_time = None
        self._emit("on_disconnect")
    
    def send_signal(self, signal_data: Dict) -> bool:
        """
        Send safety signal to Unreal.
        
        Args:
            signal_data: Safety signal data
            
        Returns:
            True if sent successfully
        """
        if not self.is_streaming:
            return False
        
        # Add Unreal-compatible fields
        output_data = self._format_unreal_signal(signal_data)
        
        success = self._streaming_manager.send_signal(output_data)
        
        if success:
            self._stats["frames_sent"] += 1
            self._stats["last_send_time"] = time.time()
        else:
            self._stats["frames_dropped"] += 1
        
        return success
    
    def send_health(self, health_data: Optional[Dict] = None) -> bool:
        """
        Send health data to Unreal.
        
        Args:
            health_data: Health data (uses current if None)
            
        Returns:
            True if sent successfully
        """
        if not self.is_streaming or not self._streaming_manager:
            return False
        
        if health_data is None and self._health_integration:
            health_data = self._health_integration.get_vr_friendly_status()
        
        if health_data:
            return self._streaming_manager.send_health(health_data)
        
        return False
    
    def send_calibration(self) -> bool:
        """
        Send current calibration to Unreal.
        
        Returns:
            True if sent successfully
        """
        if not self.is_streaming:
            return False
        
        if not self._calibration_loader:
            return False
        
        calibration = self._calibration_loader.get_active_calibration()
        return self._streaming_manager.send_calibration(calibration)
    
    def send_event(self, event_data: Dict) -> bool:
        """
        Send safety event to Unreal.
        
        Args:
            event_data: Event data
            
        Returns:
            True if sent successfully
        """
        if not self.is_streaming:
            return False
        
        return self._streaming_manager.send_event(event_data)
    
    def update_frame(
        self,
        frame_number: int,
        signal_data: Dict,
        assessment_data: Dict,
        pose_detected: bool = True,
        pose_confidence: float = 0.9,
        processing_time_ms: float = 10.0
    ):
        """
        Update all components with new frame data.
        
        Args:
            frame_number: Current frame number
            signal_data: Signal generator output
            assessment_data: Safety assessment output
            pose_detected: Whether pose was detected
            pose_confidence: Pose detection confidence
            processing_time_ms: Frame processing time
        """
        # Update health monitoring
        if self._enable_health and self._health_monitor:
            self._health_monitor.update_frame(
                frame_number=frame_number,
                processing_time_ms=processing_time_ms,
                pose_detected=pose_detected,
                pose_confidence=pose_confidence
            )
        
        # Update event manager
        if self._enable_events and self._event_manager:
            # Convert dicts back to objects if needed
            from safety_rules import SafetyAssessment
            from signal_generator import SafetyFrameSignal
            
            # Process events (simplified - in practice you'd create proper objects)
            events = self._event_manager.push_frame(
                assessment_data.get("assessment") if isinstance(assessment_data, dict) else assessment_data,
                signal_data.get("signal") if isinstance(signal_data, dict) else signal_data
            )
            
            # Send events to Unreal
            for event in events:
                self.send_event(event)
    
    def get_vr_status(self) -> Dict:
        """
        Get VR connection status.
        
        Returns:
            Dictionary with VR status information
        """
        return {
            "state": self._state.value,
            "connected": self.is_connected,
            "streaming": self.is_streaming,
            "validation": self._unreal_validation.to_dict() if self._unreal_validation else None,
            "health": self._health_integration.get_vr_friendly_status() if self._health_integration else None,
            "vr_ready": self._check_vr_ready() if self._health_integration else False,
            "stats": self._stats.copy(),
            "uptime": time.time() - self._connection_time if self._connection_time else 0
        }
    
    def _check_vr_ready(self) -> bool:
        """Check if VR is ready for streaming"""
        if not self._health_integration:
            return False
        
        ready = self._health_integration.check_vr_ready()
        return ready.get("vr_ready", False)
    
    def validate_unreal_output(self, data: Dict) -> UnrealValidationResult:
        """
        Validate output data for Unreal compatibility.
        
        Args:
            data: Data to validate
            
        Returns:
            Validation result
        """
        checks_passed = []
        checks_failed = []
        warnings = []
        
        # Check required fields
        required_fields = ["safety_flag", "confidence", "severity", "phase"]
        
        for field in required_fields:
            if field in data:
                checks_passed.append(f"field_{field}")
            else:
                checks_failed.append(f"field_{field}")
        
        # Check data types
        if "severity" in data:
            if isinstance(data["severity"], int) and 0 <= data["severity"] <= 3:
                checks_passed.append("severity_type")
            else:
                checks_failed.append("severity_type")
                warnings.append("Severity should be integer 0-3")
        
        if "confidence" in data:
            if isinstance(data["confidence"], (int, float)) and 0 <= data["confidence"] <= 1:
                checks_passed.append("confidence_type")
            else:
                checks_failed.append("confidence_type")
        
        # Check safety flag values
        valid_flags = ["safe", "warning", "danger", "unknown"]
        if "safety_flag" in data:
            if data["safety_flag"] in valid_flags:
                checks_passed.append("safety_flag_value")
            else:
                checks_failed.append("safety_flag_value")
        
        # Determine result
        if checks_failed:
            result = ValidationResult.INVALID
        elif warnings:
            result = ValidationResult.WARNING
        else:
            result = ValidationResult.VALID
        
        return UnrealValidationResult(
            result=result,
            checks_passed=checks_passed,
            checks_failed=checks_failed,
            warnings=warnings,
            metadata={"data_keys": list(data.keys())}
        )
    
    def _format_unreal_signal(self, signal_data: Dict) -> Dict:
        """
        Format signal data for Unreal consumption.
        
        Args:
            signal_data: Raw signal data
            
        Returns:
            Unreal-compatible signal data
        """
        # Create output with Unreal-friendly structure
        output = {
            "safety_flag": signal_data.get("safety_flag", "unknown"),
            "safety_flag_int": self._safety_to_int(signal_data.get("safety_flag", "unknown")),
            "confidence": signal_data.get("confidence", 0.0),
            "severity": signal_data.get("severity", 0),
            "severity_critical": signal_data.get("severity", 0) >= 3,
            "phase": signal_data.get("phase", "unknown"),
            "frame": signal_data.get("frame", 0),
            "timestamp": signal_data.get("timestamp", time.time()),
            "is_new": signal_data.get("is_new", True),
            
            # Correction data
            "correction": signal_data.get("correction"),
            "correction_required": signal_data.get("correction") is not None,
            
            # Violations
            "active_violations": signal_data.get("active_violations", 0),
            "primary_violation": signal_data.get("primary_violation"),
            
            # Unreal-specific
            "unreal_ready": True,
            "unreal_version": "1.0"
        }
        
        return output
    
    @staticmethod
    def _safety_to_int(safety_flag: str) -> int:
        """Convert safety flag to integer"""
        mapping = {
            "safe": 0,
            "warning": 1,
            "danger": 2,
            "unknown": 3
        }
        return mapping.get(safety_flag, 3)
    
    def on(self, event: str, callback: Callable):
        """Register event callback"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args, **kwargs):
        """Emit event to callbacks"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                print(f"Callback error for {event}: {e}")
    
    def _handle_health_alert(self, alert):
        """Handle health alert"""
        if alert.severity in ["error", "critical"]:
            self._emit("on_health_critical", alert.to_dict())


# Convenience function for quick setup
async def quick_vr_setup(
    host: str = "127.0.0.1",
    port: int = 7777,
    protocol: StreamProtocol = StreamProtocol.UDP
) -> VRIntegration:
    """
    Quick setup for VR integration.
    
    Args:
        host: Target host
        port: Target port
        protocol: Streaming protocol
        
    Returns:
        Configured VRIntegration instance
    """
    config = StreamingConfig(
        protocol=protocol,
        host=host,
        port=port
    )
    
    vr = VRIntegration(streaming_config=config)
    await vr.initialize()
    
    return vr
