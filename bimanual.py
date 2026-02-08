#!/usr/bin/env python3
"""Bimanual SO-101 environment scaffold for handover A -> B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import mujoco
import numpy as np

MODEL_PATH: Final[Path] = Path(__file__).parent / "models" / "so101" / "bimanual.xml"
JOINT_ORDER: Final[tuple[str, ...]] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


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
    arms: int = 2
    joints_per_arm: int = 5

    @property
    def action_dim(self) -> int:
        return self.arms * (self.joints_per_arm + 1)

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
    object_xy_jitter_m: float = 0.025
    object_yaw_jitter_rad: float = 0.45
    b_center_distance_m: float | None = None
    b_center_heading_rad: float | None = None
    b_radius_m: float = 0.1016  # 4 inches
    b_yaw_jitter_rad: float = 0.175
    control_latency_ticks: tuple[int, int] = (0, 2)
    deadband_rad: float = 0.01
    joint_noise_std_rad: float = 0.003


DEFAULT_TASK: Final[TaskSpec] = TaskSpec()
DEFAULT_CONTROL: Final[ControlSpec] = ControlSpec()
DEFAULT_RANDOMIZATION: Final[RandomizationSpec] = RandomizationSpec()


DEFAULT_DELTA_LIMITS: Final[np.ndarray] = np.array(
    [
        # Left arm
        0.05,
        0.05,
        0.05,
        0.07,
        0.09,
        0.06,
        # Right arm
        0.05,
        0.05,
        0.05,
        0.07,
        0.09,
        0.06,
    ],
    dtype=np.float32,
)


def apply_action_with_clipping(
    action: np.ndarray, prev_target: np.ndarray, delta_limits: np.ndarray
) -> np.ndarray:
    """Applies joint-delta action with per-joint clipping."""
    if action.shape != prev_target.shape or action.shape != delta_limits.shape:
        raise ValueError("action, prev_target, and delta_limits must have same shape")
    clipped_delta = np.clip(action, -delta_limits, delta_limits)
    return prev_target + clipped_delta


class ActionProcessor:
    """Buffers chunked actions and returns one clipped target per control step."""

    def __init__(self, control_spec: ControlSpec, delta_limits: np.ndarray):
        expected_dim = control_spec.action_dim
        if delta_limits.shape != (expected_dim,):
            raise ValueError(f"delta_limits shape must be ({expected_dim},)")
        self.control_spec = control_spec
        self.delta_limits = delta_limits.astype(np.float32, copy=True)
        self.prev_target = np.zeros((expected_dim,), dtype=np.float32)
        self._chunk_buffer: list[np.ndarray] = []

    def reset(self) -> None:
        self.prev_target.fill(0.0)
        self._chunk_buffer.clear()

    def set_chunk(self, chunk_actions: np.ndarray) -> None:
        if chunk_actions.shape != (self.control_spec.chunk_k, self.control_spec.action_dim):
            raise ValueError(
                "chunk_actions must have shape "
                f"({self.control_spec.chunk_k}, {self.control_spec.action_dim})"
            )
        self._chunk_buffer = [a.astype(np.float32, copy=True) for a in chunk_actions]

    def ensure_chunk_from_single_action(self, action: np.ndarray) -> None:
        if action.shape != (self.control_spec.action_dim,):
            raise ValueError(f"action must have shape ({self.control_spec.action_dim},)")
        repeated = np.repeat(action[np.newaxis, :], self.control_spec.chunk_k, axis=0)
        self.set_chunk(repeated)

    def apply_next(self) -> np.ndarray:
        if not self._chunk_buffer:
            raise RuntimeError("No chunk set. Provide chunk actions or single action first.")
        action = self._chunk_buffer.pop(0)
        target = apply_action_with_clipping(action, self.prev_target, self.delta_limits)
        self.prev_target = target
        return target


class BimanualSO101Env:
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

        self.max_control_steps = int(task.episode_seconds * control.control_hz)
        self.success_hold_steps = int(task.success_hold_seconds * control.control_hz)

        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_{j}")
                for j in JOINT_ORDER
            ]
            + [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_{j}")
                for j in JOINT_ORDER
            ],
            dtype=np.int32,
        )
        if np.any(self.actuator_ids < 0):
            raise ValueError("Missing one or more expected left/right actuators")

        self.ctrl_low = self.model.actuator_ctrlrange[self.actuator_ids, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[self.actuator_ids, 1]

        self.joint_qpos_ids = np.array(
            [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"left_{j}")
                ]
                for j in JOINT_ORDER
            ]
            + [
                self.model.jnt_qposadr[
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"right_{j}")
                ]
                for j in JOINT_ORDER
            ],
            dtype=np.int32,
        )

        block_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "block_free")
        self.block_qpos_adr = int(self.model.jnt_qposadr[block_joint_id])
        self.block_qvel_adr = int(self.model.jnt_dofadr[block_joint_id])
        self.block_body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "block"))
        self.block_geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "block_geom"))
        self.goal_site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_zone"))
        self.handover_site_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "handover_zone")
        )
        self.left_base_body_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_base")
        )
        self.right_base_body_id = int(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_base")
        )

        self._base_block_mass = float(self.model.body_mass[self.block_body_id])
        self._base_block_friction = self.model.geom_friction[self.block_geom_id].copy()
        self._left_base_pos_nominal = self.model.body_pos[self.left_base_body_id].copy()
        self._right_base_pos_nominal = self.model.body_pos[self.right_base_body_id].copy()
        self._right_base_quat_nominal = self.model.body_quat[self.right_base_body_id].copy()
        delta_xy = self._right_base_pos_nominal[:2] - self._left_base_pos_nominal[:2]
        self._ab_nominal_distance = float(np.linalg.norm(delta_xy))
        self._ab_nominal_heading = float(np.arctan2(delta_xy[1], delta_xy[0]))
        self._right_base_nominal_yaw = float(
            2.0 * np.arctan2(self._right_base_quat_nominal[3], self._right_base_quat_nominal[0])
        )

        self._rng = np.random.default_rng(0)
        self._control_step = 0
        self._goal_hold_counter = 0
        self._latency_ticks = (0, 0)
        self._latency_buffers: list[list[np.ndarray]] = [[], []]
        self._last_applied = np.zeros((self.control.action_dim,), dtype=np.float32)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._control_step = 0
        self._goal_hold_counter = 0
        self._latency_buffers = [[], []]
        self._last_applied.fill(0.0)
        self.processor.reset()

        mujoco.mj_resetData(self.model, self.data)

        home = np.array([0.0, 0.45, 0.9, -0.65, 0.0, 0.55] * 2, dtype=np.float64)
        self.data.qpos[self.joint_qpos_ids] = home
        self.data.ctrl[self.actuator_ids] = home

        self._sample_b_base_pose()
        self._apply_randomization_scales()
        self._sample_block_pose()
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
            "control_step": self._control_step,
            "max_control_steps": self.max_control_steps,
        }
        return obs, reward, done, info

    def _sample_block_pose(self) -> None:
        center = np.array([-0.12, -0.10], dtype=np.float64)
        xy = center + self._rng.uniform(
            low=-self.randomization.object_xy_jitter_m,
            high=self.randomization.object_xy_jitter_m,
            size=(2,),
        )
        z = 0.08
        yaw = self._rng.uniform(
            -self.randomization.object_yaw_jitter_rad,
            self.randomization.object_yaw_jitter_rad,
        )

        q_adr = self.block_qpos_adr
        self.data.qpos[q_adr : q_adr + 3] = np.array([xy[0], xy[1], z], dtype=np.float64)
        self.data.qpos[q_adr + 3 : q_adr + 7] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float64
        )
        self.data.qvel[self.block_qvel_adr : self.block_qvel_adr + 6] = 0.0

    def _sample_b_base_pose(self) -> None:
        # Arm A stays fixed; Arm B base is sampled in a circle around a center offset from A.
        center_distance = (
            self.randomization.b_center_distance_m
            if self.randomization.b_center_distance_m is not None
            else self._ab_nominal_distance
        )
        center_heading = (
            self.randomization.b_center_heading_rad
            if self.randomization.b_center_heading_rad is not None
            else self._ab_nominal_heading
        )
        center_xy = self._left_base_pos_nominal[:2] + center_distance * np.array(
            [np.cos(center_heading), np.sin(center_heading)],
            dtype=np.float64,
        )

        # Uniform sample inside disk area.
        phi = self._rng.uniform(0.0, 2.0 * np.pi)
        r = self.randomization.b_radius_m * np.sqrt(self._rng.uniform(0.0, 1.0))
        sampled_xy = center_xy + r * np.array([np.cos(phi), np.sin(phi)], dtype=np.float64)

        pos = self._right_base_pos_nominal.copy()
        pos[0] = sampled_xy[0]
        pos[1] = sampled_xy[1]
        self.model.body_pos[self.right_base_body_id] = pos

        yaw = self._right_base_nominal_yaw + self._rng.uniform(
            -self.randomization.b_yaw_jitter_rad,
            self.randomization.b_yaw_jitter_rad,
        )
        self.model.body_quat[self.right_base_body_id] = np.array(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
            dtype=np.float64,
        )

    def _sample_latency(self) -> None:
        low, high = self.randomization.control_latency_ticks
        self._latency_ticks = (
            int(self._rng.integers(low, high + 1)),
            int(self._rng.integers(low, high + 1)),
        )

    def _apply_deadband(self, target: np.ndarray) -> np.ndarray:
        delta = target - self._last_applied
        mask = np.abs(delta) < self.randomization.deadband_rad
        delta = np.where(mask, 0.0, delta)
        return self._last_applied + delta

    def _apply_latency(self, target: np.ndarray) -> np.ndarray:
        out = np.zeros_like(target)
        for i, (start, end) in enumerate(((0, 6), (6, 12))):
            arm_target = target[start:end].copy()
            self._latency_buffers[i].append(arm_target)
            delay = self._latency_ticks[i]
            if len(self._latency_buffers[i]) <= delay:
                out[start:end] = self._last_applied[start:end]
            else:
                out[start:end] = self._latency_buffers[i].pop(0)
        self._last_applied = out
        return out

    def _observation(self) -> np.ndarray:
        qpos = self.data.qpos[self.joint_qpos_ids]
        qvel = self.data.qvel[: self.model.nv]
        block = self.data.qpos[self.block_qpos_adr : self.block_qpos_adr + 7]

        # Add observation noise as part of per-arm randomization.
        noisy_qpos = qpos + self._rng.normal(
            0.0, self.randomization.joint_noise_std_rad, size=qpos.shape
        )

        return np.concatenate([noisy_qpos, qvel, block], axis=0).astype(np.float32)

    def _reward_and_events(self, target: np.ndarray) -> tuple[float, bool, bool]:
        block_pos = self.data.xpos[self.block_body_id]
        goal_pos = self.data.site_xpos[self.goal_site_id]
        handover_pos = self.data.site_xpos[self.handover_site_id]

        dist_goal = float(np.linalg.norm(block_pos - goal_pos))
        dist_handover = float(np.linalg.norm(block_pos - handover_pos))

        in_goal = dist_goal < self.task.success_radius_m
        self._goal_hold_counter = self._goal_hold_counter + 1 if in_goal else 0
        success = self._goal_hold_counter >= self.success_hold_steps

        dropped = bool(block_pos[2] < 0.025)

        action_mag = float(np.linalg.norm(target))
        action_delta = float(np.linalg.norm(target - self.processor.prev_target))

        reward = 0.0
        reward += -0.8 * dist_goal
        reward += -0.2 * dist_handover
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
        mass = self._base_block_mass * self._rng.uniform(
            1.0 + (mass_lo - 1.0) * strength,
            1.0 + (mass_hi - 1.0) * strength,
        )
        self.model.body_mass[self.block_body_id] = mass

        fric_lo, fric_hi = self.randomization.friction_scale
        scale = self._rng.uniform(
            1.0 + (fric_lo - 1.0) * strength,
            1.0 + (fric_hi - 1.0) * strength,
        )
        self.model.geom_friction[self.block_geom_id] = self._base_block_friction * scale


def rollout_random_policy(episodes: int = 2, seed: int = 0) -> None:
    env = BimanualSO101Env()
    rng = np.random.default_rng(seed)

    for ep in range(episodes):
        _ = env.reset(seed=seed + ep)
        done = False
        total_reward = 0.0
        steps = 0

        while not done:
            chunk = rng.uniform(-0.1, 0.1, size=(env.control.chunk_k, env.control.action_dim)).astype(
                np.float32
            )
            for _ in range(env.control.chunk_k):
                _, reward, done, info = env.step(chunk)
                total_reward += reward
                steps += 1
                if done:
                    break

        print(
            f"episode={ep} steps={steps} reward={total_reward:.3f} "
            f"success={info['success']} dropped={info['dropped']}"
        )


def main() -> None:
    print("SO-101 bimanual env scaffold")
    print(f"model: {MODEL_PATH}")
    print(f"defaults: {DEFAULT_TASK}, {DEFAULT_CONTROL}, rand={DEFAULT_RANDOMIZATION.level}")
    rollout_random_policy(episodes=2, seed=0)


if __name__ == "__main__":
    main()
