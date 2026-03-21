"""Run policy inference using the source-installed LeRobot (main branch).

Constructs a RecordConfig programmatically and calls the record() function
directly, bypassing CLI arg parsing.
"""

from __future__ import annotations

from functools import wraps

from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
from lerobot.scripts.lerobot_record import DatasetRecordConfig, RecordConfig, record

# ── Runtime settings ────────────────────────────────────────────────
NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 30
TASK_DESCRIPTION = "Block move"

HF_MODEL_ID = "ThaJpo/hf_act_recordpolicy2"
HF_DATASET_ID = "ThaJpo/eval_random-for-policy-main-env"

FRONT_CAMERA_INDEX = 0
TOP_CAMERA_INDEX = 1
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

FOLLOWER_PORT = "/dev/cu.usbmodem5AE60836341"

# ── Monkey-patch camera timeout ─────────────────────────────────────
_original_async_read = OpenCVCamera.async_read


@wraps(_original_async_read)
def _async_read_with_longer_timeout(self, timeout_ms: float = 500):
    return _original_async_read(self, timeout_ms=timeout_ms)


OpenCVCamera.async_read = _async_read_with_longer_timeout

# ── Build configs ───────────────────────────────────────────────────
robot_cfg = SOFollowerRobotConfig(
    id="follower",
    port=FOLLOWER_PORT,
    cameras={
        "front": OpenCVCameraConfig(
            index_or_path=FRONT_CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=FPS,
        ),
        "top": OpenCVCameraConfig(
            index_or_path=TOP_CAMERA_INDEX,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT,
            fps=FPS,
        ),
    },
)

dataset_cfg = DatasetRecordConfig(
    repo_id=HF_DATASET_ID,
    single_task=TASK_DESCRIPTION,
    fps=FPS,
    episode_time_s=EPISODE_TIME_SEC,
    num_episodes=NUM_EPISODES,
    video=True,
    push_to_hub=False,
)

policy_cfg = PreTrainedConfig.from_pretrained(HF_MODEL_ID)
policy_cfg.pretrained_path = HF_MODEL_ID

cfg = RecordConfig(
    robot=robot_cfg,
    dataset=dataset_cfg,
    policy=policy_cfg,
    display_data=False,
    play_sounds=True,
)


def main() -> None:
    # Call the unwrapped record() to avoid CLI parsing
    record.__wrapped__(cfg)


if __name__ == "__main__":
    main()
