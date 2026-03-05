import cv2
import numpy as np
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

FPS = 30

configs = {
    "front": OpenCVCameraConfig(index_or_path=1, width=1920, height=1080, fps=FPS, color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION),
    "top": OpenCVCameraConfig(index_or_path=0, width=1920, height=1080, fps=FPS, color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION),
}

cameras = {name: OpenCVCamera(cfg) for name, cfg in configs.items()}

for name, cam in cameras.items():
    print(f"Connecting {name} camera...")
    cam.connect()

try:
    while True:
        frames = {}
        for name, cam in cameras.items():
            frame = cam.async_read(timeout_ms=200)
            frames[name] = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Stack frames side by side
        combined = np.hstack(list(frames.values()))

        # Resize for display so it fits on screen (two 1920x1080 = 3840 wide)
        display = cv2.resize(combined, (1920, 540))
        cv2.imshow("Dual Camera Feed (press q to quit)", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cv2.destroyAllWindows()
    for cam in cameras.values():
        cam.disconnect()