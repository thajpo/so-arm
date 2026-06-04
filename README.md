# SO-ARM

SO-101 robotics workspace for sim-to-real manipulation experiments. The repo
combines a MuJoCo/MJX simulation path, hardware sanity checks, LeRobot data
collection scripts, and policy-inference run scripts for a real SO-101 arm.

This is the project-of-record for the hardware side of the MLRE portfolio. Its
job is to show that robot-learning work was grounded in physical setup,
instrumentation, configuration, and failure checks rather than only notebooks.

## What It Proves

- SO-101 simulation setup with MuJoCo/MJX and a fast visual-only model.
- Typed hardware configuration for machine-specific ports and cameras.
- Camera and USB motor-controller probes for real hardware bringup.
- LeRobot dataset recording flow for SO-101 leader/follower teleoperation.
- Policy execution script that downloads a trained ACT policy, patches runtime
  config for local inference, records evaluation episodes, and writes datasets.
- CI-backed repository-contract, hardware-config, camera-wall, and motor USB
  parsing tests.

## Setup

The current runtime is pinned to Python 3.12 so the LeRobot/JAX stack remains
stable:

```bash
rm -rf .venv uv.lock
uv python install 3.12
uv venv --python 3.12
source .venv/bin/activate
uv pip install --python .venv/bin/python -e .
```

For Linux ROCm runs, replace the default Torch build with ROCm wheels:

```bash
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/rocm7.1 \
  --force-reinstall torch torchvision
```

Verify the runtime without letting `uv` resync the environment:

```bash
uv run --no-sync python -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

Create a local hardware config:

```bash
cp hardware.example.toml hardware.toml
```

Edit `hardware.toml` for the current machine's serial ports and camera devices.
You can also point to a different file:

```bash
SO_ARM_HARDWARE_CONFIG=/path/to/hardware.toml uv run --no-sync python src/record.py
```

## Fast Checks

Run the dependency-light CI-facing tests:

```bash
python -m unittest \
  tests/test_repo_contract.py \
  tests/test_motor_usb.py \
  -v
```

After rebuilding the Python 3.12 project environment from the setup section,
run the hardware/config utility tests inside that environment:

```bash
uv run --no-sync python -m unittest \
  tests/test_hardware_config.py \
  tests/test_camera_wall.py \
  -v
```

Run lint locally:

```bash
ruff check src hardware_tests tests sim motor_usb.py
```

## Hardware Config

Example:

```toml
[ports]
follower = "/dev/ttyACM0"
leader = "/dev/ttyACM1"

[cameras.front]
device = 0
width = 640
height = 480
fps = 30
fourcc = "MJPG"
backend = "v4l2"

[cameras.top]
device = 1
width = 640
height = 480
fps = 30
fourcc = "MJPG"
backend = "v4l2"
```

The loader fails fast when required fields are missing and is shared by
recording, policy execution, camera test, and camera-wall tools.

## Simulation

Test MJX simulation:

```bash
uv run --no-sync python sim/test_so101.py
```

Visualize the arm:

```bash
uv run --no-sync python sim/view_so101.py
```

The SO-101 model is adapted from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100):

- 6 DOF position-controlled arm;
- visual meshes for rendering;
- collision geometry reduced for fast MJX iteration.

## Hardware Probes

Inspect the USB motor controller:

```bash
uv run --no-sync python hardware_tests/motor_usb.py list
uv run --no-sync python hardware_tests/motor_usb.py monitor --port /dev/cu.usbmodem5AE60836341 --baud 1000000 --seconds 10
uv run --no-sync python hardware_tests/motor_usb.py send --port /dev/cu.usbmodem5AE60836341 --baud 1000000 --text "MOTOR 120"
uv run --no-sync python hardware_tests/motor_usb.py send --port /dev/cu.usbmodem5AE60836341 --baud 1000000 --hex-payload "FF 01 00"
```

Display configured or detected cameras in one tiled view:

```bash
uv run --no-sync python src/camera_wall.py
uv run --no-sync python src/camera_wall.py --scan
```

Run the LeRobot camera feed check:

```bash
uv run --no-sync python hardware_tests/camera_test.py
```

## Data Collection And Policy Runs

Record teleoperated SO-101 episodes:

```bash
uv run --no-sync python src/record.py
```

Run a trained policy and record the evaluation episode:

```bash
uv run --no-sync python src/run_policy.py
```

These scripts intentionally keep task timing, Hugging Face model/dataset IDs,
and local policy patching visible. Hardware-specific values should live in
`hardware.toml`.

## Repo Map

- `sim/`: MJX/MuJoCo simulation and visualization entrypoints.
- `hardware_tests/`: serial motor utility and camera stress probes.
- `src/hardware_config.py`: TOML config loader and LeRobot camera config builder.
- `src/camera_wall.py`: tiled camera viewer for setup/debugging.
- `src/record.py`: SO-101 teleoperation recording flow.
- `src/run_policy.py`: local ACT policy inference and episode recording flow.
- `tests/`: lightweight contract, config, camera utility, and motor utility
  tests for hosted CI.
- `docs/reviewer-runbook.md`: no-hardware and robot-attached reviewer commands.
- `docs/run-report-template.md`: template for the real SO-101 result artifact.
- `experiments/lerobot-main-env/`: vendored/experimental LeRobot environment;
  useful as historical context, but not the primary original-code surface.

## Portfolio Roadmap

1. Add a short run report for the SO-101 pick-and-place result: task definition,
   number of demonstrations, evaluation protocol, success/failure counts, and
   the exact policy artifact used.
2. Add one GIF or image sequence of teleoperation/data collection and one GIF of
   policy inference.
3. Add import smoke tests for `src/record.py` and `src/run_policy.py` with
   LeRobot/camera/Hugging Face dependencies mocked.
4. Decide what to do with large tracked artifacts such as model weights and
   document where reviewers should inspect them.
5. Separate or document vendored LeRobot code so reviewers can quickly identify
   original code and experiment-specific patches.
