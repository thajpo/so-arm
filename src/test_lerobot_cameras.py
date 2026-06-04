from __future__ import annotations

import argparse
import time

import cv2
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

from hardware_config import build_camera_configs, load_hardware_config


CAMERA_NAMES = ("front", "top")
DEFAULT_READS = 30


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test configured LeRobot OpenCV cameras without touching robot serial ports."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional hardware config path. Defaults to hardware.toml or SO_ARM_HARDWARE_CONFIG.",
    )
    parser.add_argument(
        "--camera",
        choices=CAMERA_NAMES,
        default=None,
        help="Test only one configured camera. Default is to test both together.",
    )
    parser.add_argument(
        "--reads",
        type=int,
        default=DEFAULT_READS,
        help="Number of frame reads per camera.",
    )
    parser.add_argument(
        "--delay-s",
        type=float,
        default=0.01,
        help="Sleep between read cycles.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Override configured width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Override configured height.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Override configured fps.",
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default=None,
        help="Optional FOURCC override, for example MJPG.",
    )
    return parser.parse_args()


def selected_camera_names(camera_name: str | None) -> tuple[str, ...]:
    if camera_name is None:
        return CAMERA_NAMES
    return (camera_name,)


def main() -> int:
    args = parse_args()
    hardware_config = load_hardware_config(args.config)
    camera_names = selected_camera_names(args.camera)
    camera_configs = build_camera_configs(hardware_config, camera_names)

    cameras: dict[str, OpenCVCamera] = {}
    try:
        for name in camera_names:
            base_config = camera_configs[name]
            config = OpenCVCameraConfig(
                index_or_path=base_config.index_or_path,
                width=args.width if args.width is not None else base_config.width,
                height=args.height if args.height is not None else base_config.height,
                fps=args.fps if args.fps is not None else base_config.fps,
                fourcc=args.fourcc if args.fourcc is not None else base_config.fourcc,
                backend=base_config.backend,
            )
            camera = OpenCVCamera(config)
            print(
                f"connecting {name}: device={config.index_or_path} "
                f"width={config.width} height={config.height} fps={config.fps} "
                f"fourcc={config.fourcc}"
            )
            camera.connect()
            cameras[name] = camera
            print(f"connected {name}: {camera}")
            if camera.videocapture is not None:
                print(
                    f"actual {name}: backend={camera.videocapture.getBackendName()} "
                    f"width={camera.videocapture.get(cv2.CAP_PROP_FRAME_WIDTH)} "
                    f"height={camera.videocapture.get(cv2.CAP_PROP_FRAME_HEIGHT)} "
                    f"fps={camera.videocapture.get(cv2.CAP_PROP_FPS)} "
                    f"fourcc={decode_fourcc(camera.videocapture.get(cv2.CAP_PROP_FOURCC))}"
                )

        print("\nreading frames...")
        successes = {name: 0 for name in camera_names}
        first_shapes = {name: None for name in camera_names}

        for _ in range(args.reads):
            for name in camera_names:
                frame = cameras[name].async_read()
                if frame is not None:
                    successes[name] += 1
                    if first_shapes[name] is None:
                        first_shapes[name] = frame.shape
            time.sleep(args.delay_s)

        print("\nsummary")
        for name in camera_names:
            configured = hardware_config.require_camera(name)
            print(
                f"{name}: device={configured.device} "
                f"good_reads={successes[name]}/{args.reads} shape={first_shapes[name]}"
            )

        if all(successes[name] == args.reads for name in camera_names):
            return 0
        return 1
    finally:
        for name, camera in cameras.items():
            camera.disconnect()
            print(f"disconnected {name}")


def decode_fourcc(value: float) -> str:
    code = int(round(value))
    chars = [chr((code >> (8 * index)) & 0xFF) for index in range(4)]
    return "".join(chars).rstrip("\x00") or "<unset>"


if __name__ == "__main__":
    raise SystemExit(main())
