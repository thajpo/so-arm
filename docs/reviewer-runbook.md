# SO-ARM Reviewer Runbook

This repo contains real-robot code, but a reviewer should be able to understand
the software surface without having the robot attached.

## No-Hardware Checks

Run the dependency-light tests:

```bash
python -m unittest \
  tests/test_repo_contract.py \
  tests/test_motor_usb.py \
  -v
```

Run lint over project-owned code:

```bash
ruff check src hardware_tests tests sim motor_usb.py
```

Compile Python sources:

```bash
python -m compileall -q src hardware_tests tests sim motor_usb.py
```

## Hardware-Config Checks

After rebuilding the Python 3.12 project environment and installing the project
dependencies, run:

```bash
uv run --no-sync python -m unittest discover -s tests -p test_hardware_config.py -v
uv run --no-sync python -m unittest discover -s tests -p test_camera_wall.py -v
```

These tests depend on the LeRobot/OpenCV environment and should be skipped if
the local environment is stale or the machine is not configured for the current
robot stack.

## Robot-Attached Commands

These require the actual SO-101 hardware, cameras, and correct `hardware.toml`:

```bash
uv run --no-sync python hardware_tests/camera_test.py
uv run --no-sync python src/camera_wall.py
uv run --no-sync python src/record.py
uv run --no-sync python src/run_policy.py
```

Do not treat failures from these commands as repo-quality failures unless the
hardware environment is known-good.

## What A Manager Should See

- Hardware configuration is externalized to `hardware.toml`.
- Camera, motor, recording, and policy execution are separate surfaces.
- Hosted CI covers pure-Python contracts and serial helper behavior.
- Real-robot results should be represented by an explicit run report and media,
  not inferred from code.
