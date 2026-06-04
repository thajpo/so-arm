from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.cameras.configs import Cv2Backends
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


DEFAULT_CONFIG_ENV_VAR = "SO_ARM_HARDWARE_CONFIG"
DEFAULT_CONFIG_FILENAME = "hardware.toml"
EXAMPLE_CONFIG_FILENAME = "hardware.example.toml"


@dataclass(frozen=True)
class CameraHardwareConfig:
    device: int | str
    width: int
    height: int
    fps: int
    fourcc: str | None = None
    backend: str | None = None


@dataclass(frozen=True)
class HardwareConfig:
    ports: dict[str, str]
    cameras: dict[str, CameraHardwareConfig]

    def require_port(self, name: str) -> str:
        try:
            return self.ports[name]
        except KeyError as exc:
            raise ValueError(f"Missing required config field: ports.{name}") from exc

    def require_camera(self, name: str) -> CameraHardwareConfig:
        try:
            return self.cameras[name]
        except KeyError as exc:
            raise ValueError(f"Missing required config field: cameras.{name}") from exc

    def require_shared_fps(self, camera_names: tuple[str, ...]) -> int:
        fps_values = {self.require_camera(name).fps for name in camera_names}
        if len(fps_values) != 1:
            joined_names = ", ".join(camera_names)
            raise ValueError(
                f"All configured cameras for [{joined_names}] must share one fps value."
            )
        return next(iter(fps_values))


def default_hardware_config_path() -> Path:
    configured_path = os.environ.get(DEFAULT_CONFIG_ENV_VAR)
    if configured_path:
        return Path(configured_path).expanduser()
    return Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_FILENAME


def example_hardware_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / EXAMPLE_CONFIG_FILENAME


def load_hardware_config(path: str | Path | None = None) -> HardwareConfig:
    config_path = Path(path).expanduser() if path is not None else default_hardware_config_path()
    if not config_path.exists():
        example_path = example_hardware_config_path()
        raise FileNotFoundError(
            f"Hardware config not found at {config_path}. "
            f"Create it from {example_path.name} or set {DEFAULT_CONFIG_ENV_VAR}."
        )

    with config_path.open("rb") as file:
        raw_config = tomllib.load(file)

    return parse_hardware_config(raw_config)


def parse_hardware_config(raw_config: dict[str, Any]) -> HardwareConfig:
    ports_raw = _require_table(raw_config, "ports")
    ports = {
        port_name: _require_type(port_value, f"ports.{port_name}", str)
        for port_name, port_value in ports_raw.items()
    }

    cameras_raw = _require_table(raw_config, "cameras")
    cameras = {
        camera_name: CameraHardwareConfig(
            device=_parse_camera_device(camera_spec, camera_name),
            width=_require_int(camera_spec, f"cameras.{camera_name}.width"),
            height=_require_int(camera_spec, f"cameras.{camera_name}.height"),
            fps=_require_int(camera_spec, f"cameras.{camera_name}.fps"),
            fourcc=_optional_fourcc(camera_spec, f"cameras.{camera_name}.fourcc"),
            backend=_optional_backend(camera_spec, f"cameras.{camera_name}.backend"),
        )
        for camera_name, camera_spec in cameras_raw.items()
    }

    if not cameras:
        raise ValueError("Missing required config field: cameras")

    return HardwareConfig(ports=ports, cameras=cameras)


def build_camera_configs(
    hardware_config: HardwareConfig, camera_names: tuple[str, ...]
) -> dict[str, OpenCVCameraConfig]:
    configs: dict[str, OpenCVCameraConfig] = {}
    for name in camera_names:
        camera = hardware_config.require_camera(name)
        config = OpenCVCameraConfig(
            index_or_path=hardware_config.require_camera(name).device,
            width=camera.width,
            height=camera.height,
            fps=camera.fps,
            fourcc=camera.fourcc,
            backend=_resolve_cv2_backend(camera.backend),
        )
        configs[name] = config
    return configs


def _parse_camera_device(camera_spec: Any, camera_name: str) -> int | str:
    camera_path = f"cameras.{camera_name}"
    if not isinstance(camera_spec, dict):
        raise ValueError(f"Missing required config field: {camera_path}")

    spec = camera_spec
    if "device" not in spec:
        raise ValueError(f"Missing required config field: {camera_path}.device")

    device = spec["device"]
    if not isinstance(device, (int, str)):
        raise ValueError(
            f"Config field {camera_path}.device must be an integer index or string path."
        )
    return device


def _require_table(raw_config: Any, path: str) -> dict[str, Any]:
    value = raw_config.get(path) if isinstance(raw_config, dict) else None
    if not isinstance(value, dict):
        raise ValueError(f"Missing required config field: {path}")
    return value


def _require_type(value: Any, path: str, expected_type: type[str]) -> str:
    if not isinstance(value, expected_type):
        raise ValueError(f"Config field {path} must be a {expected_type.__name__}.")
    return value


def _require_int(raw_config: dict[str, Any], path: str) -> int:
    value = raw_config.get(path.rsplit(".", maxsplit=1)[-1])
    if not isinstance(value, int):
        raise ValueError(f"Config field {path} must be an integer.")
    return value


def _optional_fourcc(raw_config: dict[str, Any], path: str) -> str | None:
    key = path.rsplit(".", maxsplit=1)[-1]
    value = raw_config.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 4:
        raise ValueError(f"Config field {path} must be a 4-character string.")
    return value


def _optional_backend(raw_config: dict[str, Any], path: str) -> str | None:
    key = path.rsplit(".", maxsplit=1)[-1]
    value = raw_config.get(key)
    if value is None:
        value = raw_config.get("force_backend")
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Config field {path} must be a string.")
    normalized = value.lower()
    if normalized not in {"any", "v4l2"}:
        raise ValueError(f"Config field {path} must be one of: any, v4l2.")
    return normalized


def _resolve_cv2_backend(value: str | None) -> Cv2Backends:
    if value is None or value == "any":
        return Cv2Backends.ANY
    if value == "v4l2":
        return Cv2Backends.V4L2
    raise ValueError(f"Unsupported backend value: {value}")
