# SO-ARM

SO-101 robot arm simulation using MuJoCo MJX on AMD ROCm GPU.

## Setup

```bash
uv sync
```

## Usage

### Test MJX simulation
```bash
uv run test_so101.py
```

### Visualize arm
```bash
uv run view_so101.py
```

### Inspect/control USB motor controller
```bash
# list serial devices and choose the controller path
uv run motor_usb.py list

# monitor incoming serial bytes for 10 seconds
uv run motor_usb.py monitor --port /dev/cu.usbmodem5AE60836341 --baud 115200 --seconds 10

# send a text command (newline appended by default)
uv run motor_usb.py send --port /dev/cu.usbmodem5AE60836341 --baud 115200 --text "MOTOR 120"

# send raw hex bytes
uv run motor_usb.py send --port /dev/cu.usbmodem5AE60836341 --baud 115200 --hex-payload "FF 01 00"
```

## Model

The SO-101 model is from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100).

- 6 DOF position-controlled arm
- Visual meshes for rendering (no collision meshes for fast MJX)
- ~7,900 physics steps/sec on GPU with 32 parallel environments

## Files

- `test_so101.py` - Test MJX physics simulation
- `view_so101.py` - Visualize arm with MuJoCo viewer
- `models/so101/scene.xml` - Main scene file
- `models/so101/so101.xml` - Robot model (visual only, no collision meshes)
- `models/so101/assets/` - STL meshes
