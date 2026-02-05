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
