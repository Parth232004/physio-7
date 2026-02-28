#!/usr/bin/env python3
"""
FastAPI server wrapper for PhysioSafe VR Safety System with real-time streaming and integrations.

Endpoints:
- GET /health                          -> simple health check
- GET /health/metrics                  -> system health metrics (fps, latency est., clients, tracker)
- POST /session/start                  -> start monitoring (mock/webcam, duration, format, UDP)
- POST /session/stop                   -> stop monitoring
- GET /status                          -> latest system status snapshot
- GET /stream                          -> Server-Sent Events stream (JSON lines of latest signal)
- WS  /ws/stream                       -> WebSocket streaming of signals (json|unreal|minimal)
- GET /events                          -> recent structured safety events
- WS  /ws/events                       -> WebSocket streaming of structured safety events
- POST /unreal/ping                    -> send a UDP test packet to UE host:port
- POST /calibration/load               -> apply runtime calibration overrides (thresholds)
- GET /calibration/active              -> get active calibration overrides

Run:
  uvicorn server:app --reload --host 0.0.0.0 --port 8000

Notes:
- Uses a background thread to run the frame loop so the API remains responsive.
- Defaults to mock tracker to work without a camera. Use webcam with {"mock": false}.
- WebSocket and SSE both read the latest signal list from PhysioSafeSystem.
- Optional UDP broadcasting for Unreal: set ue_udp_host/ue_udp_port on session start.
- Structured safety events are produced by SafetyEventManager and exposed over HTTP/WS.
"""

import json
import socket
import threading
import time
from typing import Dict, Optional, Set, List

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from main import PhysioSafeSystem
from safety_event_manager import SafetyEventManager

app = FastAPI(title="PhysioSafe API", version="1.2.0")


class StartSessionRequest(BaseModel):
    mock: bool = True
    camera: int = 0
    format: str = "json"  # json|unreal|vr|minimal (controls console/storage output)
    duration: float = 0.0  # 0 means unlimited
    verbose: bool = False
    no_cooldown: bool = False
    no_dedup: bool = False
    # Disable demo mode by default in service to avoid console overlays
    no_demo: bool = True
    config: str = "config.json"
    # Unreal/UDP streaming options
    ue_udp_host: Optional[str] = None
    ue_udp_port: Optional[int] = None
    ue_format: str = "unreal"  # unreal|minimal|json


class CalibrationPayload(BaseModel):
    # Structure: { "shoulder": {"flexion": {"safe_max": 120, "warning_max": 100, "danger_threshold": 130}}, ... }
    thresholds: Dict[str, Dict[str, Dict[str, float]]]


# Global runtime state
_system: Optional[PhysioSafeSystem] = None
_loop_thread: Optional[threading.Thread] = None
_stop_flag: bool = False
_state_lock = threading.Lock()

# WebSocket clients
_ws_clients: Set[WebSocket] = set()
_ws_lock = threading.Lock()

# Event streaming clients
_ws_event_clients: Set[WebSocket] = set()
_ws_event_lock = threading.Lock()

# UDP broadcast state
_udp_thread: Optional[threading.Thread] = None
_udp_cfg: Dict[str, Optional[object]] = {"host": None, "port": None, "format": "unreal"}
_last_udp_frame_sent: int = -1

# Structured safety events
_events_manager: SafetyEventManager = SafetyEventManager()
_last_event_frame: int = -1

# Calibration overrides (for introspection)
_active_calibration: Dict[str, Dict[str, Dict[str, float]]] = {}


def _run_loop(duration: Optional[float]):
    global _system, _stop_flag, _last_event_frame
    assert _system is not None
    start = time.time()
    _system.is_running = True
    while True:
        if _stop_flag:
            break
        if duration and duration > 0 and (time.time() - start) > duration:
            break
        # Process frame (safe)
        _system._process_frame_safe()
        # Push into safety event manager if a new frame arrived
        if _system.signals:
            sig = _system.signals[-1]
            if sig.frame_number != _last_event_frame:
                _last_event_frame = sig.frame_number
                # Guard if assessments list is shorter
                assess = _system.assessments[-1] if _system.assessments else None
                if assess is not None:
                    _events_manager.push_frame(assess, sig)
        time.sleep(0.0)  # yield
    _system._shutdown()


def _format_for(fmt: str, signal):
    # fmt options: json, unreal, minimal
    fmt = (fmt or "json").lower()
    if fmt == "unreal":
        try:
            # Use engine's Unreal formatter to stay consistent
            return _system.signal_engine._format_unreal(signal)  # type: ignore
        except Exception:
            return {
                "safety_flag": signal.safety_flag,
                "confidence": signal.confidence,
                "severity": signal.severity,
                "action_required": signal.severity >= 2,
                "command_code": "STOP_NOW" if signal.safety_flag == "danger" else ("CORRECT_POSITION" if signal.safety_flag == "warning" else "CONTINUE"),
                "phase": signal.phase,
            }
    if fmt == "minimal":
        try:
            return _system.signal_engine._format_minimal(signal)  # type: ignore
        except Exception:
            return {"s": signal.safety_flag[:1].upper(), "v": signal.severity, "p": signal.phase[:1].upper()}
    # default json
    return signal.to_dict()


def _udp_broadcast_loop():
    global _system, _stop_flag, _udp_cfg, _last_udp_frame_sent
    host = _udp_cfg.get("host")
    port = _udp_cfg.get("port")
    fmt = _udp_cfg.get("format", "unreal")
    if not host or not port:
        return
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        while True:
            if _stop_flag or not _system or not _system.is_running:
                break
            if _system.signals:
                sig = _system.signals[-1]
                if sig.frame_number != _last_udp_frame_sent:
                    _last_udp_frame_sent = sig.frame_number
                    payload = _format_for(fmt, sig)
                    try:
                        data = json.dumps(payload).encode("utf-8")
                        sock.sendto(data, (host, int(port)))
                    except Exception:
                        pass
            time.sleep(0.02)
    finally:
        try:
            sock.close()
        except Exception:
            pass


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/health/metrics")
def health_metrics():
    global _system, _ws_clients
    if not _system:
        return {"is_running": False}
    # Approximate FPS using available timing
    duration = max(1e-6, (time.time() - _system.start_time) if _system.start_time else 0.0)
    fps = (_system.frame_count / duration) if duration > 0 else 0.0
    last = _system.signals[-1] if _system.signals else None
    # Processing times
    avg_ms = round((sum(_system.processing_times) / len(_system.processing_times) * 1000.0), 2) if _system.processing_times else 0.0
    last_ms = round((_system.processing_times[-1] * 1000.0), 2) if _system.processing_times else 0.0
    # Tracker stats if available
    tracker_stats = {}
    try:
        if hasattr(_system, 'tracker') and hasattr(_system.tracker, 'get_statistics'):
            tracker_stats = _system.tracker.get_statistics()
    except Exception:
        tracker_stats = {}
    return {
        "is_running": _system.is_running,
        "frames": _system.frame_count,
        "uptime_sec": round(duration, 3),
        "approx_fps": round(fps, 2),
        "avg_processing_ms": avg_ms,
        "last_processing_ms": last_ms,
        "last_signal": last.to_dict() if last else None,
        "ws_clients": len(_ws_clients),
        "udp_enabled": bool(_udp_cfg.get("host") and _udp_cfg.get("port")),
        "tracker": tracker_stats,
    }


@app.post("/session/start")
def start_session(req: StartSessionRequest):
    global _system, _loop_thread, _stop_flag, _udp_thread, _udp_cfg, _last_udp_frame_sent, _events_manager, _last_event_frame
    with _state_lock:
        if _loop_thread and _loop_thread.is_alive():
            raise HTTPException(status_code=409, detail="Session already running")
        # Create system
        _system = PhysioSafeSystem(
            use_mock_tracker=req.mock,
            camera_index=req.camera,
            output_format=req.format,
            verbose=req.verbose,
            cooldown_enabled=not req.no_cooldown,
            deduplication_enabled=not req.no_dedup,
            demo_mode=not req.no_demo,
            config_path=req.config,
        )
        if not _system.initialize():
            _system = None
            raise HTTPException(status_code=500, detail="Failed to initialize PhysioSafeSystem")
        # Reset events on new session
        _events_manager = SafetyEventManager()
        _last_event_frame = -1
        # Start background frame loop
        _stop_flag = False
        duration = None if req.duration == 0 else req.duration
        _loop_thread = threading.Thread(target=_run_loop, args=(duration,), daemon=True)
        _loop_thread.start()
        # Configure UDP if requested
        _udp_cfg = {"host": req.ue_udp_host, "port": req.ue_udp_port, "format": (req.ue_format or "unreal")}
        _last_udp_frame_sent = -1
        if req.ue_udp_host and req.ue_udp_port:
            _udp_thread = threading.Thread(target=_udp_broadcast_loop, daemon=True)
            _udp_thread.start()
        return {
            "message": "session started",
            "mock": req.mock,
            "format": req.format,
            "udp_enabled": bool(req.ue_udp_host and req.ue_udp_port),
            "ws_endpoint": "/ws/stream",
            "ws_events_endpoint": "/ws/events",
            "sse_endpoint": "/stream",
        }


@app.post("/session/stop")
def stop_session():
    global _system, _loop_thread, _stop_flag
    with _state_lock:
        if not _loop_thread or not _loop_thread.is_alive():
            return {"message": "no active session"}
        _stop_flag = True
    if _loop_thread:
        _loop_thread.join(timeout=5)
    with _state_lock:
        _loop_thread = None
        _stop_flag = False
    return {"message": "session stopped"}


@app.get("/status")
def status():
    global _system
    if not _system:
        return {"is_running": False}
    return _system.get_current_status()


@app.get("/stream")
def stream():
    global _system
    if not _system:
        raise HTTPException(status_code=400, detail="No active session. Start one via /session/start")

    def event_source():
        last_frame = -1
        try:
            while True:
                if not _system.is_running:
                    break
                if _system.signals and _system.signals[-1].frame_number != last_frame:
                    last_frame = _system.signals[-1].frame_number
                    data = _system.signals[-1].to_json()
                    yield f"data: {data}\n\n"
                time.sleep(0.05)
        except GeneratorExit:
            return

    return StreamingResponse(event_source(), media_type="text/event-stream")


@app.websocket("/ws/stream")
async def ws_stream(websocket: WebSocket):
    global _system, _ws_clients
    await websocket.accept()
    with _ws_lock:
        _ws_clients.add(websocket)
    # handshake: attempt to read optional client hello
    preferred_format = "unreal"
    try:
        try:
            msg = await websocket.receive_text()
            try:
                obj = json.loads(msg)
                if isinstance(obj, dict) and obj.get("type") == "hello":
                    # choose format if provided
                    accepts = obj.get("accepts") or []
                    if isinstance(accepts, list) and accepts:
                        # pick first supported
                        for f in ["unreal", "minimal", "json"]:
                            if f in accepts:
                                preferred_format = f
                                break
                await websocket.send_text(json.dumps({"type": "welcome", "schema_version": "1.0", "format": preferred_format}))
            except Exception:
                await websocket.send_text(json.dumps({"type": "welcome", "schema_version": "1.0", "format": preferred_format}))
        except Exception:
            # no initial message, still proceed
            await websocket.send_text(json.dumps({"type": "welcome", "schema_version": "1.0", "format": preferred_format}))

        last_frame = -1
        while True:
            if not _system or not _system.is_running:
                await asyncio_sleep_nonblocking(0.05)
                continue
            if _system.signals and _system.signals[-1].frame_number != last_frame:
                sig = _system.signals[-1]
                last_frame = sig.frame_number
                payload = _format_for(preferred_format, sig)
                await websocket.send_text(json.dumps(payload))
            await asyncio_sleep_nonblocking(0.02)
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)


@app.get("/events")
def events(limit: int = Query(50, ge=1, le=200)):
    # return recent events (up to limit)
    return {"events": _events_manager.history(limit)}


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    global _ws_event_clients
    await websocket.accept()
    with _ws_event_lock:
        _ws_event_clients.add(websocket)
    try:
        # Start from current end of history; stream new events as they appear
        last_len = len(_events_manager.history())
        while True:
            cur_hist: List[dict] = _events_manager.history()
            if len(cur_hist) > last_len:
                new_items = cur_hist[last_len:]
                for ev in new_items:
                    await websocket.send_text(json.dumps(ev))
                last_len = len(cur_hist)
            await asyncio_sleep_nonblocking(0.1)
    except WebSocketDisconnect:
        pass
    finally:
        with _ws_event_lock:
            if websocket in _ws_event_clients:
                _ws_event_clients.remove(websocket)


async def asyncio_sleep_nonblocking(delay: float):
    # Local helper to avoid importing asyncio directly in handler body
    try:
        import asyncio
        await asyncio.sleep(delay)
    except Exception:
        time.sleep(delay)


@app.post("/unreal/ping")
def unreal_ping(host: str, port: int, message: str = "UE_PING", fmt: str = "unreal"):
    payload = {"type": "ping", "msg": message, "ts": time.time()}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(1.0)
        sock.sendto(json.dumps(payload).encode("utf-8"), (host, int(port)))
        sock.close()
        return {"sent": True, "payload": payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"UDP send failed: {e}")


@app.post("/calibration/load")
def calibration_load(payload: CalibrationPayload):
    global _system, _active_calibration
    overrides = payload.thresholds or {}
    if not _system:
        raise HTTPException(status_code=400, detail="No active system. Start a session first.")
    applied = _apply_calibration_overrides(_system, overrides)
    _active_calibration = overrides if applied else {}
    return {"applied": applied, "overrides": _active_calibration}


@app.get("/calibration/active")
def calibration_active():
    return {"overrides": _active_calibration}


def _apply_calibration_overrides(system: PhysioSafeSystem, overrides: Dict[str, Dict[str, Dict[str, float]]]) -> bool:
    """Apply threshold overrides directly on SafetyRules for both left and right sides."""
    try:
        rules = system.safety_rules
        for joint_group, movements in overrides.items():
            for movement, vals in movements.items():
                for side in ["left", "right"]:
                    key = f"{joint_group}_{side}_{movement}"
                    th = rules.thresholds.get(key)
                    if not th:
                        continue
                    if "safe_max" in vals:
                        th.safe_max = float(vals["safe_max"])  # type: ignore
                    if "warning_max" in vals:
                        th.warning_max = float(vals["warning_max"])  # type: ignore
                    if "danger_threshold" in vals:
                        th.danger_max = float(vals["danger_threshold"])  # type: ignore
        return True
    except Exception:
        return False


# Root route
@app.get("/")
def root():
    return {
        "service": "PhysioSafe API",
        "version": app.version,
        "endpoints": [
            "/health", "/health/metrics", "/session/start", "/session/stop", "/status", "/stream", "/ws/stream",
            "/events", "/ws/events",
            "/unreal/ping", "/calibration/load", "/calibration/active",
        ],
    }
