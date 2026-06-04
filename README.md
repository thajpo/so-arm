# SO-ARM

SO-101 robot arm simulation using MuJoCo MJX on AMD ROCm GPU.

## Setup

```bash
rm -rf .venv uv.lock
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python -e .

# Replace the default torch build in this venv with ROCm wheels.
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.1 \
  --force-reinstall torch torchvision

# Verify without letting uv resync the environment.
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
cp hardware.example.toml hardware.toml
```

Edit `hardware.toml` for the current machine's serial ports and camera devices. You can also point to a different file with `SO_ARM_HARDWARE_CONFIG=/path/to/hardware.toml`.

## Usage

### Test MJX simulation
```bash
uv run --no-sync test_so101.py
```

### Visualize arm
```bash
uv run --no-sync view_so101.py
```

### Inspect/control USB motor controller
```bash
# list serial devices and choose the controller path
uv run --no-sync motor_usb.py list

# monitor incoming serial bytes for 10 seconds
uv run --no-sync motor_usb.py monitor --port /dev/cu.usbmodem5AE60836341 --baud 115200 --seconds 10

# send a text command (newline appended by default)
uv run --no-sync motor_usb.py send --port /dev/cu.usbmodem5AE60836341 --baud 115200 --text "MOTOR 120"

# send raw hex bytes
uv run --no-sync motor_usb.py send --port /dev/cu.usbmodem5AE60836341 --baud 115200 --hex-payload "FF 01 00"
```

### Hardware config
```toml
[ports]
follower = "/dev/ttyACM0"
leader = "/dev/ttyACM1"

[cameras.front]
device = 0
width = 640
height = 480
fps = 30

[cameras.top]
device = 1
width = 640
height = 480
fps = 30
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
