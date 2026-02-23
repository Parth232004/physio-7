# PhysioSafe VR Safety System - Demo Mode (Hardened)

Real-time physiotherapy exercise safety monitoring system with VR-ready signal output.

## What Runs

- **Main System**: Real-time pose tracking → safety assessment → VR signals
- **Demo Mode**: Hardened single-exercise demo with predictable flow
- **Mock Mode**: Simulated pose data for testing without webcam

## How to Run

### Quick Test (60 seconds)
```bash
python run_demo.py --mock --quick-test
```

### Full Demo (10 minutes with mock)
```bash
python run_demo.py --mock --duration 600
```

### Webcam Demo (5 minutes)
```bash
python run_demo.py --webcam --duration 300
```

### Direct Run (main.py)
```bash
python main.py --mock --duration 60
```

## Demo Flow

| Minute | Phase | Description |
|--------|-------|-------------|
| 1 | Good Form | Safe range shoulder flexion |
| 2 | Caution | Slight over-flexion → WARNING |
| 3 | Correction | Safe correction back to range |
| 4 | STOP | Force danger scenario |
| 5 | Recovery | Return to safe position |
| 6+ | Summary | Session statistics |

## Features

- **DEMO_MODE**: Single exercise lock (shoulder_flexion)
- **Safety Override**: RED banner on danger
- **Crash Guards**: No pose, camera disconnect, zero division handling
- **Visual Overlay**: Clear status with 2s cooldown
- **Session Logging**: JSON export for analysis

## Known Limitations

- Single exercise only (no multi-exercise switching)
- Mock mode uses synthetic pose data (no real user tracking)
- Requires webcam/MediaPipe for actual pose detection
