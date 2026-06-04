from pathlib import Path
import sys

import cv2
import numpy as np
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hardware_config import build_camera_configs, load_hardware_config
from lerobot_camera_compat import apply_opencv_linux_compat_patch


CAMERA_NAMES = ("front", "top")
apply_opencv_linux_compat_patch()

hardware_config = load_hardware_config()
configs = build_camera_configs(hardware_config, CAMERA_NAMES)
display_width = sum(config.width for config in configs.values())
display_height = max(config.height for config in configs.values())

for config in configs.values():
    config.color_mode = ColorMode.RGB
    config.rotation = Cv2Rotation.NO_ROTATION

cameras = {name: OpenCVCamera(cfg) for name, cfg in configs.items()}

for name in CAMERA_NAMES:
    camera_device = hardware_config.require_camera(name).device
    print(f"Connecting {name} camera at {camera_device}...")
    cam = cameras[name]
    cam.connect()

try:
    while True:
        frames = {}
        for name, cam in cameras.items():
            frame = cam.async_read(timeout_ms=200)
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Stack frames side by side
        combined = np.hstack(list(frames.values()))

        display = cv2.resize(combined, (display_width, display_height))
        cv2.imshow("Dual Camera Feed (press q to quit)", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    for cam in cameras.values():
        cam.disconnect()
