"""
Live Signal Streaming Layer for Unreal/VR Real-Time Consumption

Provides multiple streaming protocols:
- WebSocket: For web-based clients and modern Unreal implementations
- UDP: Low-latency for Unreal Engine network replication
- TCP: Reliable ordered delivery for critical safety signals

Supports:
- Real-time signal streaming
- Connection handshake with VR clients
- Health monitoring integration
- Calibration data synchronization
"""

import json
import socket
import struct
import threading
import time
import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Union
from collections import deque


class StreamProtocol(Enum):
    """Supported streaming protocols"""
    WEBSOCKET = "websocket"
    UDP = "udp"
    TCP = "tcp"


class ConnectionState(Enum):
    """Connection state for streaming"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    ERROR = "error"


@dataclass
class StreamMessage:
    """Standard message format for streaming"""
    message_type: str  # "signal", "health", "calibration", "event", "heartbeat", "ack"
    payload: Dict[str, Any]
    timestamp: float
    sequence_number: int = 0
    client_id: Optional[str] = None
    
    def to_bytes(self) -> bytes:
        """Serialize message to bytes for UDP/TCP"""
        json_str = json.dumps(self.to_dict())
        return json_str.encode('utf-8')
    
    def to_dict(self) -> Dict:
        return {
            "type": self.message_type,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "sequence_number": self.sequence_number,
            "client_id": self.client_id
        }
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'StreamMessage':
        """Deserialize message from bytes"""
        obj = json.loads(data.decode('utf-8'))
        return cls(
            message_type=obj.get("type", "unknown"),
            payload=obj.get("payload", {}),
            timestamp=obj.get("timestamp", 0),
            sequence_number=obj.get("sequence_number", 0),
            client_id=obj.get("client_id")
        )


@dataclass
class StreamingConfig:
    """Configuration for streaming layer"""
    protocol: StreamProtocol = StreamProtocol.UDP
    host: str = "127.0.0.1"
    port: int = 7777
    buffer_size: int = 65535
    heartbeat_interval: float = 1.0  # seconds
    reconnect_delay: float = 2.0
    max_reconnect_attempts: int = 5
    enable_compression: bool = False
    unreal_compatible: bool = True  # Use Unreal-friendly format
    
    # Health monitoring
    health_check_interval: float = 0.5
    max_latency_ms: float = 100.0
    
    # Message queue
    message_queue_size: int = 100


class BaseStreamClient(ABC):
    """Abstract base class for streaming clients"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.connection_state = ConnectionState.DISCONNECTED
        self.sequence_number = 0
        self._running = False
        self._callbacks: Dict[str, List[Callable]] = {
            "on_connect": [],
            "on_disconnect": [],
            "on_message": [],
            "on_error": [],
            "on_health_update": [],
            "on_calibration": []
        }
        self._message_queue: deque = deque(maxlen=config.message_queue_size)
        self._client_id: Optional[str] = None
        self._server_info: Dict = {}
        
    @abstractmethod
    def connect(self) -> bool:
        """Establish connection"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close connection"""
        pass
    
    @abstractmethod
    def send(self, message: StreamMessage) -> bool:
        """Send message"""
        pass
    
    @abstractmethod
    def receive(self, timeout: float = 1.0) -> Optional[StreamMessage]:
        """Receive message"""
        pass
    
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
    
    def _next_sequence(self) -> int:
        """Get next sequence number"""
        self.sequence_number += 1
        return self.sequence_number
    
    def create_message(self, msg_type: str, payload: Dict) -> StreamMessage:
        """Create a stream message"""
        return StreamMessage(
            message_type=msg_type,
            payload=payload,
            timestamp=time.time(),
            sequence_number=self._next_sequence(),
            client_id=self._client_id
        )


class UDPStreamClient(BaseStreamClient):
    """UDP-based streaming client for low-latency Unreal integration"""
    
    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self._socket: Optional[socket.socket] = None
        self._server_address: Optional[tuple] = None
        
    def connect(self) -> bool:
        try:
            self.connection_state = ConnectionState.CONNECTING
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, self.config.buffer_size)
            self._server_address = (self.config.host, self.config.port)
                       # Send handshake
            handshake = self.create_message("handshake", {
                "protocol_version": "1.0",
                "client_type": "physiosafe",
                "capabilities": ["signal", "health", "calibration", "event"]
            })
            self.send(handshake)
            
            # Try to receive acknowledgment (with timeout)
            self._socket.settimeout(2.0)
            try:
                ack_data, _ = self._socket.recvfrom(self.config.buffer_size)
                ack = StreamMessage.from_bytes(ack_data)
                if ack.message_type == "ack" and ack.payload.get("accepted"):
                    self._client_id = ack.payload.get("client_id")
                    self._server_info = ack.payload.get("server_info", {})
                    self.connection_state = ConnectionState.AUTHENTICATED
                    self._emit("on_connect", self._client_id)
                    return True
            except socket.timeout:
                # No response - still allow connection for one-way streaming
                self.connection_state = ConnectionState.CONNECTED
                self._client_id = f"client_{int(time.time())}"
                return True
                
        except Exception as e:
            self.connection_state = ConnectionState.ERROR
            self._emit("on_error", str(e))
            return False
        return False
    
    def disconnect(self):
        if self._socket:
            try:
                goodbye = self.create_message("disconnect", {"reason": "client_disconnect"})
                self.send(goodbye)
            except:
                pass
            self._socket.close()
            self._socket = None
        self.connection_state = ConnectionState.DISCONNECTED
        self._emit("on_disconnect")
    
    def send(self, message: StreamMessage) -> bool:
        if not self._socket or self.connection_state == ConnectionState.DISCONNECTED:
            return False
        try:
            data = message.to_bytes()
            self._socket.sendto(data, self._server_address)
            return True
        except Exception as e:
            self._emit("on_error", str(e))
            return False
    
    def receive(self, timeout: float = 1.0) -> Optional[StreamMessage]:
        if not self._socket:
            return None
        try:
            self._socket.settimeout(timeout)
            data, addr = self._socket.recvfrom(self.config.buffer_size)
            return StreamMessage.from_bytes(data)
        except socket.timeout:
            return None
        except Exception as e:
            self._emit("on_error", str(e))
            return None


class TCPStreamClient(BaseStreamClient):
    """TCP-based streaming client for reliable ordered delivery"""
    
    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self._socket: Optional[socket.socket] = None
        
    def connect(self) -> bool:
        try:
            self.connection_state = ConnectionState.CONNECTING
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            self._socket.connect((self.config.host, self.config.port))
            
            # Send handshake
            handshake = self.create_message("handshake", {
                "protocol_version": "1.0",
                "client_type": "physiosafe",
                "capabilities": ["signal", "health", "calibration", "event"]
            })
            self._send_raw(handshake.to_bytes())
            
            # Receive acknowledgment
            self._socket.settimeout(5.0)
            ack_data = self._recv_raw()
            if ack_data:
                ack = StreamMessage.from_bytes(ack_data)
                if ack.message_type == "ack" and ack.payload.get("accepted"):
                    self._client_id = ack.payload.get("client_id")
                    self._server_info = ack.payload.get("server_info", {})
                    self.connection_state = ConnectionState.AUTHENTICATED
                    self._emit("on_connect", self._client_id)
                    return True
                    
        except Exception as e:
            self.connection_state = ConnectionState.ERROR
            self._emit("on_error", str(e))
        return False
    
    def disconnect(self):
        if self._socket:
            try:
                goodbye = self.create_message("disconnect", {"reason": "client_disconnect"})
                self._send_raw(goodbye.to_bytes())
            except:
                pass
            self._socket.close()
            self._socket = None
        self.connection_state = ConnectionState.DISCONNECTED
        self._emit("on_disconnect")
    
    def send(self, message: StreamMessage) -> bool:
        if not self._socket or self.connection_state == ConnectionState.DISCONNECTED:
            return False
        try:
            self._send_raw(message.to_bytes())
            return True
        except Exception as e:
            self._emit("on_error", str(e))
            return False
    
    def receive(self, timeout: float = 1.0) -> Optional[StreamMessage]:
        if not self._socket:
            return None
        try:
            self._socket.settimeout(timeout)
            data = self._recv_raw()
            if data:
                return StreamMessage.from_bytes(data)
        except socket.timeout:
            return None
        except Exception as e:
            self._emit("on_error", str(e))
        return None
    
    def _send_raw(self, data: bytes):
        """Send length-prefixed data"""
        length = struct.pack('!I', len(data))
        self._socket.sendall(length + data)
    
    def _recv_raw(self) -> Optional[bytes]:
        """Receive length-prefixed data"""
        length_data = self._recv_exact(4)
        if not length_data:
            return None
        length = struct.unpack('!I', length_data)[0]
        return self._recv_exact(length)
    
    def _recv_exact(self, n: int) -> Optional[bytes]:
        """Receive exact number of bytes"""
        data = b''
        while len(data) < n:
            chunk = self._socket.recv(n - len(data))
            if not chunk:
                return None
            data += chunk
        return data


class WebSocketStreamClient(BaseStreamClient):
    """WebSocket-based streaming client (requires websocket-client library)"""
    
    def __init__(self, config: StreamingConfig):
        super().__init__(config)
        self._ws = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        
    def connect(self) -> bool:
        try:
            import websocket
            ws_url = f"ws://{self.config.host}:{self.config.port}"
            self._ws = websocket.create_connection(ws_url, timeout=5.0)
            
            # Send handshake
            handshake = self.create_message("handshake", {
                "protocol_version": "1.0",
                "client_type": "physiosafe",
                "capabilities": ["signal", "health", "calibration", "event"]
            })
            self._ws.send(handshake.to_json())
            
            # Receive acknowledgment
            response = self._ws.recv()
            ack = StreamMessage.from_bytes(response.encode('utf-8'))
            if ack.message_type == "ack" and ack.payload.get("accepted"):
                self._client_id = ack.payload.get("client_id")
                self._server_info = ack.payload.get("server_info", {})
                self.connection_state = ConnectionState.AUTHENTICATED
                self._emit("on_connect", self._client_id)
                return True
                
        except ImportError:
            print("websocket-client not installed. Using TCP fallback.")
            return False
        except Exception as e:
            self.connection_state = ConnectionState.ERROR
            self._emit("on_error", str(e))
        return False
    
    def disconnect(self):
        if self._ws:
            try:
                self._ws.close()
            except:
                pass
            self._ws = None
        self.connection_state = ConnectionState.DISCONNECTED
        self._emit("on_disconnect")
    
    def send(self, message: StreamMessage) -> bool:
        if not self._ws:
            return False
        try:
            self._ws.send(message.to_json())
            return True
        except Exception as e:
            self._emit("on_error", str(e))
            return False
    
    def receive(self, timeout: float = 1.0) -> Optional[StreamMessage]:
        if not self._ws:
            return None
        try:
            self._ws.settimeout(timeout)
            data = self._ws.recv()
            if isinstance(data, str):
                data = data.encode('utf-8')
            return StreamMessage.from_bytes(data)
        except:
            return None


class LiveStreamingManager:
    """
    Main manager for live signal streaming to Unreal/VR.
    
    Features:
    - Multiple protocol support (UDP, TCP, WebSocket)
    - Automatic reconnection
    - Message queuing
    - Health monitoring integration
    - Calibration synchronization
    """
    
    def __init__(self, config: Optional[StreamingConfig] = None):
        self.config = config or StreamingConfig()
        self._client: Optional[BaseStreamClient] = None
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        
        # Statistics
        self._stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "last_send_time": 0,
            "last_receive_time": 0,
            "avg_latency_ms": 0,
            "total_latency": 0,
            "latency_samples": 0
        }
        
        # Calibration data
        self._calibration: Dict[str, Any] = {}
        
        # Health monitor callback
        self._health_callback: Optional[Callable] = None
        
    def create_client(self, protocol: Optional[StreamProtocol] = None) -> BaseStreamClient:
        """Create a streaming client for the specified protocol"""
        protocol = protocol or self.config.protocol
        
        if protocol == StreamProtocol.UDP:
            self._client = UDPStreamClient(self.config)
        elif protocol == StreamProtocol.TCP:
            self._client = TCPStreamClient(self.config)
        elif protocol == StreamProtocol.WEBSOCKET:
            self._client = WebSocketStreamClient(self.config)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
        
        return self._client
    
    def connect(self, protocol: Optional[StreamProtocol] = None) -> bool:
        """Connect to streaming server"""
        if not self._client:
            self.create_client(protocol)
        
        return self._client.connect()
    
    def disconnect(self):
        """Disconnect from streaming server"""
        self._running = False
        if self._client:
            self._client.disconnect()
    
    def is_connected(self) -> bool:
        """Check if currently connected"""
        return self._client and self._client.connection_state in [
            ConnectionState.CONNECTED, 
            ConnectionState.AUTHENTICATED
        ]
    
    def send_signal(self, signal_data: Dict) -> bool:
        """Send safety signal to Unreal/VR"""
        if not self.is_connected():
            return False
        
        message = self._client.create_message("signal", signal_data)
        
        # Add Unreal-specific fields
        if self.config.unreal_compatible:
            message.payload["unreal_ready"] = True
            message.payload["safety_flag_int"] = self._safety_to_int(signal_data.get("safety_flag", "unknown"))
        
        success = self._client.send(message)
        
        if success:
            self._stats["messages_sent"] += 1
            self._stats["last_send_time"] = time.time()
        
        return success
    
    def send_health(self, health_data: Dict) -> bool:
        """Send health monitoring data"""
        if not self.is_connected():
            return False
        
        message = self._client.create_message("health", health_data)
        return self._client.send(message)
    
    def send_calibration(self, calibration_data: Dict) -> bool:
        """Send calibration data"""
        if not self.is_connected():
            return False
        
        message = self._client.create_message("calibration", calibration_data)
        return self._client.send(message)
    
    def send_event(self, event_data: Dict) -> bool:
        """Send safety event"""
        if not self.is_connected():
            return False
        
        message = self._client.create_message("event", event_data)
        return self._client.send(message)
    
    def request_calibration(self) -> bool:
        """Request calibration data from server"""
        if not self.is_connected():
            return False
        
        message = self._client.create_message("request_calibration", {
            "request_type": "calibration_data"
        })
        return self._client.send(message)
    
    def validate_connection(self) -> Dict:
        """Validate connection and return status"""
        return {
            "connected": self.is_connected(),
            "state": self._client.connection_state.value if self._client else "none",
            "client_id": self._client._client_id if self._client else None,
            "server_info": self._client._server_info if self._client else {},
            "stats": self.get_statistics()
        }
    
    def get_statistics(self) -> Dict:
        """Get streaming statistics"""
        stats = self._stats.copy()
        if stats["latency_samples"] > 0:
            stats["avg_latency_ms"] = stats["total_latency"] / stats["latency_samples"]
        return stats
    
    def set_health_callback(self, callback: Callable):
        """Set callback for health updates"""
        self._health_callback = callback
    
    def start_async_receiver(self):
        """Start background thread for receiving messages"""
        self._running = True
        self._worker_thread = threading.Thread(target=self._receive_loop, daemon=True)
        self._worker_thread.start()
    
    def _receive_loop(self):
        """Background loop for receiving messages"""
        while self._running:
            if not self.is_connected():
                time.sleep(0.1)
                continue
            
            message = self._client.receive(timeout=0.5)
            if message:
                self._stats["messages_received"] += 1
                self._stats["last_receive_time"] = time.time()
                
                # Calculate latency
                latency = (time.time() - message.timestamp) * 1000
                self._stats["total_latency"] += latency
                self._stats["latency_samples"] += 1
                
                # Handle message types
                self._handle_message(message)
    
    def _handle_message(self, message: StreamMessage):
        """Handle received message"""
        if message.message_type == "health" and self._health_callback:
            self._health_callback(message.payload)
        elif message.message_type == "calibration":
            self._calibration = message.payload
            self._client._emit("on_calibration", message.payload)
        elif message.message_type == "ack":
            # Handshake acknowledged
            pass
        elif message.message_type == "error":
            self._stats["errors"] += 1
            self._client._emit("on_error", message.payload)
    
    @staticmethod
    def _safety_to_int(safety_flag: str) -> int:
        """Convert safety flag to integer for Unreal"""
        mapping = {
            "safe": 0,
            "warning": 1,
            "danger": 2,
            "unknown": 3
        }
        return mapping.get(safety_flag, 3)


# Server-side implementation for Unreal to receive data
class StreamingServer:
    """Server for receiving data from PhysioSafe (for testing/validation)"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self._socket: Optional[socket.socket] = None
        self._running = False
        self._clients: Dict[str, BaseStreamClient] = {}
        self._callbacks: Dict[str, List[Callable]] = {
            "on_client_connect": [],
            "on_client_disconnect": [],
            "on_signal": [],
            "on_health": [],
            "on_calibration": []
        }
    
    def start(self, protocol: StreamProtocol = StreamProtocol.UDP):
        """Start streaming server"""
        self._running = True
        
        if protocol == StreamProtocol.UDP:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.config.host, self.config.port))
            self._socket.settimeout(1.0)
        elif protocol == StreamProtocol.TCP:
            self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._socket.bind((self.config.host, self.config.port))
            self._socket.listen(5)
        
        print(f"Streaming server started on {self.config.host}:{self.config.port}")
        
        while self._running:
            try:
                if protocol == StreamProtocol.UDP:
                    self._handle_udp()
                elif protocol == StreamProtocol.TCP:
                    self._handle_tcp()
            except Exception as e:
                if self._running:
                    print(f"Server error: {e}")
    
    def stop(self):
        """Stop server"""
        self._running = False
        if self._socket:
            self._socket.close()
    
    def _handle_udp(self):
        """Handle UDP client"""
        try:
            data, addr = self._socket.recvfrom(self.config.buffer_size)
            message = StreamMessage.from_bytes(data)
            self._process_message(message, str(addr))
        except socket.timeout:
            pass
    
    def _handle_tcp(self):
        """Handle TCP client"""
        try:
            client_socket, addr = self._socket.accept()
            client_socket.settimeout(5.0)
            
            # Receive handshake
            length_data = client_socket.recv(4)
            if length_data:
                length = struct.unpack('!I', length_data)[0]
                data = client_socket.recv(length)
                message = StreamMessage.from_bytes(data)
                
                if message.message_type == "handshake":
                    # Send acknowledgment
                    ack = StreamMessage(
                        message_type="ack",
                        payload={
                            "accepted": True,
                            "client_id": f"client_{addr[1]}",
                            "server_info": {"version": "1.0"}
                        },
                        timestamp=time.time()
                    )
                    client_socket.sendall(ack.to_bytes())
                    
                    self._emit("on_client_connect", str(addr))
                    
                    # Receive loop
                    while self._running:
                        try:
                            length_data = client_socket.recv(4)
                            if not length_data:
                                break
                            length = struct.unpack('!I', length_data)[0]
                            data = client_socket.recv(length)
                            message = StreamMessage.from_bytes(data)
                            self._process_message(message, str(addr))
                        except socket.timeout:
                            continue
                        except:
                            break
                    
                    self._emit("on_client_disconnect", str(addr))
            client_socket.close()
        except Exception as e:
            print(f"TCP handler error: {e}")
    
    def _process_message(self, message: StreamMessage, client_id: str):
        """Process received message"""
        if message.message_type == "signal":
            self._emit("on_signal", message.payload)
        elif message.message_type == "health":
            self._emit("on_health", message.payload)
        elif message.message_type == "calibration":
            self._emit("on_calibration", message.payload)
    
    def on(self, event: str, callback: Callable):
        """Register event callback"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _emit(self, event: str, *args):
        """Emit event"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                print(f"Callback error: {e}")
