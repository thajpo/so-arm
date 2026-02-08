#!/usr/bin/env python3
"""Single-arm v0 sim2real environment scaffold for SO-101."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mujoco
import numpy as np

MODEL_PATH: Final[Path] = Path(__file__).parent / "models" / "so101" / "single_arm_task.xml"
JOINT_ORDER: Final[tuple[str, ...]] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Bidirectional transport zones in world coordinates.
ZONE_A_XY: Final[np.ndarray] = np.array([-0.16, -0.08], dtype=np.float64)
ZONE_B_XY: Final[np.ndarray] = np.array([0.16, 0.08], dtype=np.float64)


@dataclass(frozen=True)
class TaskSpec:
    episode_seconds: float = 8.0
    success_hold_seconds: float = 0.5
    success_radius_m: float = 0.04


@dataclass(frozen=True)
class ControlSpec:
    sim_hz: int = 500
    control_hz: int = 20
    chunk_k: int = 4
    action_mode: str = "joint_delta"
    joints: int = 6

    @property
    def action_dim(self) -> int:
        return self.joints

    @property
    def frame_skip(self) -> int:
        if self.sim_hz % self.control_hz != 0:
            raise ValueError("sim_hz must be divisible by control_hz")
        return self.sim_hz // self.control_hz


@dataclass(frozen=True)
class RandomizationSpec:
    level: str = "med"
    object_mass_scale: tuple[float, float] = (0.8, 1.2)
    friction_scale: tuple[float, float] = (0.7, 1.3)
    spawn_xy_jitter_m: float = 0.03
    spawn_yaw_jitter_rad: float = 0.5
    control_latency_ticks: tuple[int, int] = (0, 2)
    deadband_rad: float = 0.01
    joint_noise_std_rad: float = 0.003


DEFAULT_TASK: Final[TaskSpec] = TaskSpec()
DEFAULT_CONTROL: Final[ControlSpec] = ControlSpec()
DEFAULT_RANDOMIZATION: Final[RandomizationSpec] = RandomizationSpec()

DEFAULT_DELTA_LIMITS: Final[np.ndarray] = np.array(
    [0.05, 0.05, 0.05, 0.07, 0.09, 0.06], dtype=np.float32
)


class ActionProcessor:
    """Converts chunked delta actions into clipped control targets."""

    def __init__(self, control: ControlSpec, delta_limits: np.ndarray):
        if delta_limits.shape != (control.action_dim,):
            raise ValueError(f"delta_limits must have shape ({control.action_dim},)")
        self.control = control
        self.delta_limits = delta_limits.astype(np.float32, copy=True)
        self.prev_target = np.zeros((control.action_dim,), dtype=np.float32)
        self._chunk_buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self.prev_target.fill(0.0)
        self._chunk_buffer.clear()

    def set_chunk(self, chunk_actions: np.ndarray) -> None:
        if chunk_actions.shape != (self.control.chunk_k, self.control.action_dim):
            raise ValueError(
                f"chunk_actions must have shape ({self.control.chunk_k}, {self.control.action_dim})"
            )
        self._chunk_buffer = [a.astype(np.float32, copy=True) for a in chunk_actions]

    def ensure_chunk_from_single_action(self, action: np.ndarray) -> None:
        if action.shape != (self.control.action_dim,):
            raise ValueError(f"action must have shape ({self.control.action_dim},)")
        chunk = np.repeat(action[np.newaxis, :], self.control.chunk_k, axis=0)
        self.set_chunk(chunk)

    def apply_next(self) -> np.ndarray:
        if not self._chunk_buffer:
            raise RuntimeError("No chunk buffered")
        raw = self._chunk_buffer.pop(0)
        clipped = np.clip(raw, -self.delta_limits, self.delta_limits)
        target = self.prev_target + clipped
        self.prev_target = target
        return target


class SingleArmSO101Env:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        task: TaskSpec = DEFAULT_TASK,
        control: ControlSpec = DEFAULT_CONTROL,
        randomization: RandomizationSpec = DEFAULT_RANDOMIZATION,
        delta_limits: np.ndarray = DEFAULT_DELTA_LIMITS,
    ):
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.control = control
        self.randomization = randomization
        self.processor = ActionProcessor(control, delta_limits)

        self.max_control_steps = int(self.task.episode_seconds * self.control.control_hz)
        self.success_hold_steps = int(self.task.success_hold_seconds * self.control.control_hz)

        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in JOINT_ORDER
            ],
            dtype=np.int32,
        )
        self.joint_qpos_ids = np.array(
            [
                self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
                for name in JOINT_ORDER
            ],
            dtype=np.int32,
        )

        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1]

        self.block_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block"))
        self.block_geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom"))
        self.block_joint_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_free"))
        self.block_qpos_adr = int(self.model.jnt_qposadr[self.block_joint_id])
        self.block_qvel_adr = int(self.model.jnt_dofadr[self.block_joint_id])

        self.source_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "source_zone"))
        self.target_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target_zone"))

        self._base_block_mass = float(self.model.body_mass[self.block_body_id])
        self._base_block_friction = self.model.geom_friction[self.block_geom_id].copy()

        self._rng = np.random.default_rng(0)
        self._control_step = 0
        self._goal_hold_counter = 0
        self._latency_ticks = 0
        self._latency_buffer: list[np.ndarray] = []
        self._last_applied = np.zeros((self.control.action_dim,), dtype=np.float32)
        self._source_xy = ZONE_A_XY.copy()
        self._target_xy = ZONE_B_XY.copy()

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._control_step = 0
        self._goal_hold_counter = 0
        self._latency_buffer = []
        self._last_applied.fill(0.0)
        self.processor.reset()

        mujoco.mj_resetData(self.model, self.data)

        # Fixed home for single-arm v0.
        home = np.array([0.0, 0.45, 0.9, -0.65, 0.0, 0.55], dtype=np.float64)
        self.data.qpos[self.joint_qpos_ids] = home
        self.data.ctrl[self.actuator_ids] = home

        self._sample_direction()
        self._apply_randomization_scales()
        self._sample_block_pose_near_source()
        self._sample_latency()

        mujoco.mj_forward(self.model, self.data)
        return self._observation()

    def step(self, action_or_chunk: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        a = np.asarray(action_or_chunk, dtype=np.float32)
        if a.ndim == 2:
            self.processor.set_chunk(a)
        elif a.ndim == 1 and not self.processor._chunk_buffer:
            self.processor.ensure_chunk_from_single_action(a)
        elif a.ndim != 1:
            raise ValueError("action_or_chunk must be shape (A,) or (K, A)")

        target = self.processor.apply_next()
        target = self._apply_deadband(target)
        target = self._apply_latency(target)
        target = np.clip(target, self.ctrl_low, self.ctrl_high)

        self.data.ctrl[self.actuator_ids] = target
        for _ in range(self.control.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self._control_step += 1
        obs = self._observation()
        reward, success, dropped = self._reward_and_events(target)
        done = success or dropped or (self._control_step >= self.max_control_steps)
        info = {
            "success": success,
            "dropped": dropped,
            "direction": "A_to_B" if np.allclose(self._source_xy, ZONE_A_XY) else "B_to_A",
            "control_step": self._control_step,
            "max_control_steps": self.max_control_steps,
        }
        return obs, reward, done, info

    def _sample_direction(self) -> None:
        if self._rng.random() < 0.5:
            self._source_xy = ZONE_A_XY.copy()
            self._target_xy = ZONE_B_XY.copy()
        else:
            self._source_xy = ZONE_B_XY.copy()
            self._target_xy = ZONE_A_XY.copy()

        src = self.model.site_pos[self.source_site_id]
        tgt = self.model.site_pos[self.target_site_id]
        src[:2] = self._source_xy
        tgt[:2] = self._target_xy

    def _sample_block_pose_near_source(self) -> None:
        xy = self._source_xy + self._rng.uniform(
            low=-self.randomization.spawn_xy_jitter_m,
            high=self.randomization.spawn_xy_jitter_m,
            size=(2,),
        )
        yaw = self._rng.uniform(
            -self.randomization.spawn_yaw_jitter_rad,
            self.randomization.spawn_yaw_jitter_rad,
        )
        z = 0.08

        q = self.block_qpos_adr
        self.data.qpos[q : q + 3] = np.array([xy[0], xy[1], z], dtype=np.float64)
        self.data.qpos[q + 3 : q + 7] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64
        )
        self.data.qvel[self.block_qvel_adr : self.block_qvel_adr + 6] = 0.0

    def _sample_latency(self) -> None:
        low, high = self.randomization.control_latency_ticks
        self._latency_ticks = int(self._rng.integers(low, high + 1))

    def _apply_deadband(self, target: np.ndarray) -> np.ndarray:
        delta = target - self._last_applied
        delta = np.where(np.abs(delta) < self.randomization.deadband_rad, 0.0, delta)
        return self._last_applied + delta

    def _apply_latency(self, target: np.ndarray) -> np.ndarray:
        self._latency_buffer.append(target.copy())
        if len(self._latency_buffer) <= self._latency_ticks:
            out = self._last_applied.copy()
        else:
            out = self._latency_buffer.pop(0)
        self._last_applied = out
        return out

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos[self.joint_qpos_ids]
        qvel = self.data.qvel[: self.model.nv]
        block = self.data.qpos[self.block_qpos_adr : self.block_qpos_adr + 7]
        target = np.array([self._target_xy[0], self._target_xy[1], 0.08], dtype=np.float64)

        noisy_qpos = qpos + self._rng.normal(0.0, self.randomization.joint_noise_std_rad, size=qpos.shape)
        return np.concatenate([noisy_qpos, qvel, block, target], axis=0).astype(np.float32)

    def _reward_and_events(self, target: np.ndarray) -> tuple[float, bool, bool]:
        block_pos = self.data.xpos[self.block_body_id]
        goal = np.array([self._target_xy[0], self._target_xy[1], 0.08], dtype=np.float64)

        dist_goal = float(np.linalg.norm(block_pos - goal))
        in_goal = dist_goal < self.task.success_radius_m
        self._goal_hold_counter = self._goal_hold_counter + 1 if in_goal else 0
        success = self._goal_hold_counter >= self.success_hold_steps

        dropped = bool(block_pos[2] < 0.025)

        # Progress reward is simple and robust for v0.
        action_mag = float(np.linalg.norm(target))
        action_delta = float(np.linalg.norm(target - self.processor.prev_target))

        reward = -0.8 * dist_goal
        reward += -0.002 * action_mag
        reward += -0.004 * action_delta
        if dropped:
            reward -= 5.0
        if success:
            reward += 10.0
        return reward, success, dropped

    def _apply_randomization_scales(self) -> None:
        strength = {"low": 0.5, "med": 1.0, "high": 1.5}.get(self.randomization.level, 1.0)

        mass_lo, mass_hi = self.randomization.object_mass_scale
        mass_scale = self._rng.uniform(
            1.0 + (mass_lo - 1.0) * strength,
            1.0 + (mass_hi - 1.0) * strength,
        )
        self.model.body_mass[self.block_body_id] = self._base_block_mass * mass_scale

        fric_lo, fric_hi = self.randomization.friction_scale
        fric_scale = self._rng.uniform(
            1.0 + (fric_lo - 1.0) * strength,
            1.0 + (fric_hi - 1.0) * strength,
        )
        self.model.geom_friction[self.block_geom_id] = self._base_block_friction * fric_scale


def quick_eval(env: SingleArmSO101Env, episodes: int = 20, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    success = 0
    drops = 0
    dir_success = {"A_to_B": 0, "B_to_A": 0}
    dir_count = {"A_to_B": 0, "B_to_A": 0}

    for ep in range(episodes):
        env.reset(seed=seed + ep)
        done = False
        info = {}
        while not done:
            chunk = rng.uniform(-0.1, 0.1, size=(env.control.chunk_k, env.control.action_dim)).astype(np.float32)
            for _ in range(env.control.chunk_k):
                _, _, done, info = env.step(chunk)
                if done:
                    break

        direction = info["direction"]
        dir_count[direction] += 1
        if info["success"]:
            success += 1
            dir_success[direction] += 1
        if info["dropped"]:
            drops += 1

    print(f"episodes={episodes}")
    print(f"success_rate={success / episodes:.3f}")
    print(f"drop_rate={drops / episodes:.3f}")
    for d in ("A_to_B", "B_to_A"):
        rate = dir_success[d] / max(1, dir_count[d])
        print(f"{d}_success={rate:.3f} (n={dir_count[d]})")


def main() -> None:
    env = SingleArmSO101Env()
    print("Single-arm SO-101 v0 env")
    print(f"model={MODEL_PATH}")
    print(f"task={DEFAULT_TASK}")
    print(f"control={DEFAULT_CONTROL}")
    print(f"randomization={DEFAULT_RANDOMIZATION.level}")

    quick_eval(env, episodes=20, seed=0)


if __name__ == "__main__":
    main()
