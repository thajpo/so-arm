from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

from hardware_config import build_camera_configs, load_hardware_config


WINDOW_NAME = "SO-ARM LeRobot Camera Wall"
CAMERA_NAMES = ("front", "top")
LABEL_BAR_HEIGHT = 28
LABEL_MARGIN = 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display configured cameras using the patched LeRobot camera path."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional hardware config path. Defaults to hardware.toml or SO_ARM_HARDWARE_CONFIG.",
    )
    return parser.parse_args()


def compute_grid(count: int) -> tuple[int, int]:
    if count < 1:
        raise ValueError("Camera count must be positive.")
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols


def tile_position(index: int, cols: int, tile_width: int, tile_height: int) -> tuple[int, int]:
    row = index // cols
    col = index % cols
    return row * tile_height, col * tile_width


def annotate_frame(frame: np.ndarray, label: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], LABEL_BAR_HEIGHT), (20, 20, 20), -1)
    cv2.putText(
        output,
        label,
        (LABEL_MARGIN, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output


def blank_frame(width: int, height: int, label: str) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        frame,
        f"offline: {label}",
        (LABEL_MARGIN, height // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return annotate_frame(frame, label)


def main() -> int:
    args = parse_args()
    hardware_config = load_hardware_config(args.config)
    camera_configs = build_camera_configs(hardware_config, CAMERA_NAMES)

    cameras: dict[str, OpenCVCamera] = {}
    tile_width = 0
    tile_height = 0

    try:
        for name in CAMERA_NAMES:
            camera = OpenCVCamera(camera_configs[name])
            print(f"connecting {name}: {camera_configs[name].index_or_path}")
            camera.connect()
            cameras[name] = camera
            if camera.videocapture is not None:
                tile_width = max(tile_width, int(round(camera.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH))))
                tile_height = max(tile_height, int(round(camera.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                print(
                    f"connected {name}: backend={camera.videocapture.getBackendName()} "
                    f"{tile_width}x{tile_height}@{camera.videocapture.get(cv2.CAP_PROP_FPS):.1f}"
                )

        rows, cols = compute_grid(len(cameras))
        canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

        while True:
            canvas.fill(0)
            for index, name in enumerate(CAMERA_NAMES):
                camera = cameras[name]
                frame = camera.async_read()
                if frame is None:
                    tile = blank_frame(tile_width, tile_height, name)
                else:
                    if frame.shape[1] != tile_width or frame.shape[0] != tile_height:
                        frame = cv2.resize(frame, (tile_width, tile_height))
                    tile = annotate_frame(frame, f"{name} [{camera.config.index_or_path}]")
                y, x = tile_position(index, cols, tile_width, tile_height)
                canvas[y : y + tile_height, x : x + tile_width] = tile

            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        return 0
    finally:
        for name, camera in cameras.items():
            camera.disconnect()
            print(f"disconnected {name}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
