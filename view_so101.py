#!/usr/bin/env python3
"""Visualize SO-101 arm with MuJoCo viewer."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "so101" / "scene.xml"

print(f"Loading model from: {MODEL_PATH}")

# Load model
mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
mj_data = mujoco.MjData(mj_model)

print(f"Model loaded: {mj_model.nq} joints, {mj_model.nu} actuators")
print(f"Joint names: {[mj_model.joint(i).name for i in range(mj_model.njnt)]}")
print()

# Demo: move through some poses
def demo_poses():
    """Generator that yields control values for demo."""
    t = 0
    while True:
        # Sinusoidal motion on each joint with different frequencies
        ctrl = np.array([
            0.5 * np.sin(t * 0.5),           # shoulder_pan
            0.3 * np.sin(t * 0.7 + 1),       # shoulder_lift
            0.5 * np.sin(t * 0.6 + 2),       # elbow_flex
            0.4 * np.sin(t * 0.8 + 3),       # wrist_flex
            0.8 * np.sin(t * 0.4 + 4),       # wrist_roll
            0.5 + 0.5 * np.sin(t * 1.0),     # gripper (0 to 1)
        ])
        t += 0.02
        yield ctrl

poses = demo_poses()

def controller(model, data):
    """Controller callback for viewer."""
    data.ctrl[:] = next(poses)

print("Starting viewer... (close window to exit)")
print("The arm will move through demo poses automatically.")

# Launch interactive viewer
with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
    while viewer.is_running():
        # Apply controls
        controller(mj_model, mj_data)
        
        # Step simulation
        mujoco.mj_step(mj_model, mj_data)
        
        # Sync viewer
        viewer.sync()
        
        # Real-time pacing
        time.sleep(mj_model.opt.timestep)
