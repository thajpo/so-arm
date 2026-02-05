#!/usr/bin/env python3
"""Test loading SO-101 arm in MJX."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

MODEL_PATH = Path(__file__).parent / "models" / "so101" / "scene.xml"

print(f"Loading model from: {MODEL_PATH}")
print(f"JAX devices: {jax.devices()}")
print()

# Load MuJoCo model
mj_model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
mj_data = mujoco.MjData(mj_model)

print(f"Model loaded: {mj_model.nq} qpos, {mj_model.nv} qvel, {mj_model.nu} actuators")
print(f"Joint names: {[mj_model.joint(i).name for i in range(mj_model.njnt)]}")
print(f"Actuator names: {[mj_model.actuator(i).name for i in range(mj_model.nu)]}")
print()

# Convert to MJX
print("Converting to MJX...")
mjx_model = mjx.put_model(mj_model)
mjx_data = mjx.put_data(mj_model, mj_data)
print("MJX model ready!")
print()

# JIT compile step function
step = jax.jit(mjx.step)

# Test single step
print("=== Test: Single step ===")
start = time.time()
mjx_data = step(mjx_model, mjx_data)
jax.block_until_ready(mjx_data)
print(f"First step (JIT compile): {time.time() - start:.4f}s")

start = time.time()
for _ in range(100):
    mjx_data = step(mjx_model, mjx_data)
jax.block_until_ready(mjx_data)
print(f"100 steps: {time.time() - start:.4f}s")
print(f"qpos: {mjx_data.qpos}")
print()

# Test parallelized simulation
print("=== Test: Parallelized simulation ===")
batch_size = 32

@jax.jit
def batch_step(model, data):
    return jax.vmap(mjx.step, in_axes=(None, 0))(model, data)

# Create batch by tiling
single_data = mjx.put_data(mj_model, mj_data)
batch_data = jax.tree_util.tree_map(
    lambda x: jnp.tile(x, (batch_size,) + (1,) * x.ndim), single_data
)

start = time.time()
batch_data = batch_step(mjx_model, batch_data)
jax.block_until_ready(batch_data)
print(f"First batch step (JIT compile): {time.time() - start:.4f}s")

start = time.time()
for _ in range(100):
    batch_data = batch_step(mjx_model, batch_data)
jax.block_until_ready(batch_data)
print(f"100 batch steps ({batch_size} envs): {time.time() - start:.4f}s")
print(f"Steps/sec: {100 * batch_size / (time.time() - start + 0.0001):.0f}")
print()

print("SO-101 loaded and running on MJX!")
