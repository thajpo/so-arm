from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from hardware_config import load_hardware_config


WINDOW_NAME = "SO-ARM Camera Wall"
DEFAULT_SCAN_LIMIT = 8
DEFAULT_TILE_WIDTH = 640
DEFAULT_TILE_HEIGHT = 480
LABEL_BAR_HEIGHT = 28
LABEL_MARGIN = 8
V4L_BY_PATH_DIR = Path("/dev/v4l/by-path")
V4L_BY_ID_DIR = Path("/dev/v4l/by-id")


@dataclass(frozen=True)
class CameraTarget:
    label: str
    device: int | str
    width: int
    height: int
    fps: int | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Display all configured or detected cameras in one tiled window."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional hardware config path. Defaults to hardware.toml or SO_ARM_HARDWARE_CONFIG.",
    )
    parser.add_argument(
        "--scan",
        action="store_true",
        help="Ignore hardware config and scan /dev/video* and integer indices.",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=DEFAULT_SCAN_LIMIT,
        help="Maximum integer indices to probe when scanning fallback devices.",
    )
    return parser.parse_args()


def load_camera_targets(config_path: str | None, scan: bool, max_cameras: int) -> list[CameraTarget]:
    if not scan:
        try:
            hardware_config = load_hardware_config(config_path)
        except FileNotFoundError:
            hardware_config = None
        else:
            return [
                CameraTarget(
                    label=f"{name} [{camera.device}]",
                    device=camera.device,
                    width=camera.width,
                    height=camera.height,
                    fps=camera.fps,
                )
                for name, camera in hardware_config.cameras.items()
            ]

    return scan_camera_targets(max_cameras=max_cameras)


def scan_camera_targets(max_cameras: int) -> list[CameraTarget]:
    by_path_targets = discover_index0_targets(V4L_BY_PATH_DIR)
    if by_path_targets:
        return by_path_targets

    by_id_targets = discover_index0_targets(V4L_BY_ID_DIR)
    if by_id_targets:
        return by_id_targets

    device_paths = sorted(
        Path("/dev").glob("video*"),
        key=lambda path: numeric_suffix(path.name),
    )
    if device_paths:
        return [
            CameraTarget(
                label=str(path),
                device=str(path),
                width=DEFAULT_TILE_WIDTH,
                height=DEFAULT_TILE_HEIGHT,
                fps=None,
            )
            for path in device_paths[::2]
        ]

    return [
        CameraTarget(
            label=f"index {index}",
            device=index,
            width=DEFAULT_TILE_WIDTH,
            height=DEFAULT_TILE_HEIGHT,
            fps=None,
        )
        for index in range(max_cameras)
    ]


def discover_index0_targets(directory: Path) -> list[CameraTarget]:
    if not directory.exists():
        return []

    symlinks = sorted(directory.iterdir(), key=lambda path: path.name)
    targets: list[CameraTarget] = []
    seen_devices: set[str] = set()
    for symlink in symlinks:
        if "video-index0" not in symlink.name:
            continue
        device_path = str(symlink.resolve())
        if device_path in seen_devices:
            continue
        targets.append(
            CameraTarget(
                label=build_scan_label(symlink, device_path),
                device=device_path,
                width=DEFAULT_TILE_WIDTH,
                height=DEFAULT_TILE_HEIGHT,
                fps=None,
            )
        )
        seen_devices.add(device_path)
    return targets


def build_scan_label(symlink: Path, device_path: str) -> str:
    if symlink.parent == V4L_BY_PATH_DIR:
        return f"{shorten_by_path_label(symlink.name)} [{device_path}]"
    return f"{symlink.name} [{device_path}]"


def shorten_by_path_label(name: str) -> str:
    match = re.search(r"usb(?:v2)?-(.+)-video-index0$", name)
    if not match:
        return name
    return f"port {match.group(1)}"


def numeric_suffix(name: str) -> int:
    match = re.search(r"(\d+)$", name)
    if match is None:
        return 1_000_000
    return int(match.group(1))


def open_capture(target: CameraTarget) -> cv2.VideoCapture | None:
    capture = cv2.VideoCapture(target.device, cv2.CAP_V4L2)
    if not capture.isOpened():
        capture.release()
        return None

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(target.width))
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(target.height))
    if target.fps is not None:
        capture.set(cv2.CAP_PROP_FPS, float(target.fps))
    return capture


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


def render_wall(targets: list[CameraTarget], captures: list[cv2.VideoCapture]) -> None:
    tile_width = max(int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))) for capture in captures) or DEFAULT_TILE_WIDTH
    tile_height = (
        max(int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))) for capture in captures) or DEFAULT_TILE_HEIGHT
    )
    rows, cols = compute_grid(len(captures))
    canvas = np.zeros((rows * tile_height, cols * tile_width, 3), dtype=np.uint8)

    while True:
        canvas.fill(0)
        for index, (capture, target) in enumerate(zip(captures, targets, strict=True)):
            ok, frame = capture.read()
            if not ok or frame is None:
                tile = blank_frame(tile_width, tile_height, target.label)
            else:
                resized = cv2.resize(frame, (tile_width, tile_height))
                tile = annotate_frame(resized, target.label)

            y, x = tile_position(index, cols, tile_width, tile_height)
            canvas[y : y + tile_height, x : x + tile_width] = tile

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break


def main() -> int:
    args = parse_args()
    targets = load_camera_targets(args.config, args.scan, args.max_cameras)

    captures: list[cv2.VideoCapture] = []
    opened_targets: list[CameraTarget] = []
    try:
        for target in targets:
            capture = open_capture(target)
            if capture is None:
                continue
            captures.append(capture)
            opened_targets.append(target)

        if not captures:
            print("No cameras opened. Check hardware.toml or rerun with --scan.")
            return 1

        print("Opened cameras:")
        for target, capture in zip(opened_targets, captures, strict=True):
            width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
            height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            fps = capture.get(cv2.CAP_PROP_FPS)
            print(f"  {target.label}: {width}x{height} @ {fps:.1f} FPS")

        render_wall(opened_targets, captures)
        return 0
    finally:
        for capture in captures:
            capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
