from __future__ import annotations

import json
import shutil
from functools import wraps
from pathlib import Path

from huggingface_hub import snapshot_download
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.record import record_loop
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower
from lerobot.utils.control_utils import init_keyboard_listener
from lerobot.utils.utils import log_say
from lerobot.utils.visualization_utils import _init_rerun

_original_async_read = OpenCVCamera.async_read


@wraps(_original_async_read)
def _async_read_with_longer_timeout(self, timeout_ms: float = 500):
    return _original_async_read(self, timeout_ms=timeout_ms)


OpenCVCamera.async_read = _async_read_with_longer_timeout

NUM_EPISODES = 1
FPS = 30
EPISODE_TIME_SEC = 30
TASK_DESCRIPTION = "Block move"

HF_MODEL_ID = "ThaJpo/hf_act_recordpolicy2"
HF_DATASET_ID = "ThaJpo/random-for-policy"

LOCAL_MODEL_ROOT = Path(__file__).resolve().parents[1] / "models"
LOCAL_MODEL_DIR = LOCAL_MODEL_ROOT / "hf_act_recordpolicy2"
PATCHED_MODEL_DIR = LOCAL_MODEL_ROOT / "hf_act_recordpolicy2_patched"


def prepare_local_policy_repo() -> Path:
    LOCAL_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    if not LOCAL_MODEL_DIR.exists():
        snapshot_download(
            repo_id=HF_MODEL_ID,
            repo_type="model",
            local_dir=str(LOCAL_MODEL_DIR),
        )

    if PATCHED_MODEL_DIR.exists():
        shutil.rmtree(PATCHED_MODEL_DIR)
    shutil.copytree(LOCAL_MODEL_DIR, PATCHED_MODEL_DIR)

    config_path = PATCHED_MODEL_DIR / "config.json"
    config = json.loads(config_path.read_text())

    config.pop("use_peft", None)
    config.pop("pretrained_path", None)
    config["device"] = "cpu"

    config_path.write_text(json.dumps(config, indent=4) + "\n")

    preprocessor_path = PATCHED_MODEL_DIR / "policy_preprocessor.json"
    if preprocessor_path.exists():
        preprocessor = json.loads(preprocessor_path.read_text())
        for step in preprocessor.get("steps", []):
            if step.get("registry_name") == "device_processor":
                step.setdefault("config", {})["device"] = "cpu"
        preprocessor_path.write_text(json.dumps(preprocessor, indent=2) + "\n")

    postprocessor_path = PATCHED_MODEL_DIR / "policy_postprocessor.json"
    if postprocessor_path.exists():
        postprocessor = json.loads(postprocessor_path.read_text())
        for step in postprocessor.get("steps", []):
            if step.get("registry_name") == "device_processor":
                step.setdefault("config", {})["device"] = "cpu"
        postprocessor_path.write_text(json.dumps(postprocessor, indent=2) + "\n")

    return PATCHED_MODEL_DIR


camera_config = {
    "front": OpenCVCameraConfig(index_or_path=0, width=640, height=480, fps=FPS),
    "top": OpenCVCameraConfig(index_or_path=1, width=640, height=480, fps=FPS),
}

robot_config = SO101FollowerConfig(
    id="follower",
    cameras=camera_config,
    port="/dev/cu.usbmodem5AE60836341",
)

model_dir = prepare_local_policy_repo()

robot = SO101Follower(robot_config)
policy = ACTPolicy.from_pretrained(str(model_dir))

action_features = hw_to_dataset_features(robot.action_features, "action")
obs_features = hw_to_dataset_features(robot.observation_features, "observation")
dataset_features = {**action_features, **obs_features}

dataset = LeRobotDataset.create(
    repo_id=HF_DATASET_ID,
    fps=FPS,
    features=dataset_features,
    robot_type=robot.name,
    use_videos=True,
    image_writer_threads=4,
)

_, events = init_keyboard_listener()
_init_rerun(session_name="policy-inference")

robot.connect()

try:
    for episode_idx in range(NUM_EPISODES):
        if events["stop_recording"]:
            break

        log_say(
            f"Running policy inference, recording eval episode "
            f"{episode_idx + 1} of {NUM_EPISODES}"
        )

        record_loop(
            robot=robot,
            events=events,
            fps=FPS,
            policy=policy,
            dataset=dataset,
            control_time_s=EPISODE_TIME_SEC,
            single_task=TASK_DESCRIPTION,
            display_data=False,
        )

        if events["rerecord_episode"]:
            log_say("Re-recording episode")
            events["rerecord_episode"] = False
            events["exit_early"] = False
            dataset.clear_episode_buffer()
            continue

        dataset.save_episode()
finally:
    log_say("Stopping policy run")
    robot.disconnect()
    dataset.push_to_hub()
