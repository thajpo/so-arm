from pathlib import Path
import sys
import unittest

from lerobot.cameras.configs import Cv2Backends


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hardware_config import build_camera_configs, parse_hardware_config


class HardwareConfigTests(unittest.TestCase):
    def test_parse_valid_config(self) -> None:
        config = parse_hardware_config(
            {
                "ports": {
                    "follower": "/dev/ttyACM0",
                    "leader": "/dev/ttyACM1",
                },
                "cameras": {
                    "front": {
                        "device": 0,
                        "width": 640,
                        "height": 480,
                        "fps": 30,
                        "fourcc": "MJPG",
                        "backend": "v4l2",
                    },
                    "top": {"device": "/dev/video2", "width": 640, "height": 480, "fps": 30},
                },
            }
        )

        self.assertEqual(config.require_port("follower"), "/dev/ttyACM0")
        self.assertEqual(config.require_camera("top").device, "/dev/video2")
        self.assertEqual(config.require_shared_fps(("front", "top")), 30)

        camera_configs = build_camera_configs(config, ("front", "top"))
        self.assertEqual(camera_configs["front"].index_or_path, 0)
        self.assertEqual(camera_configs["top"].index_or_path, "/dev/video2")
        self.assertEqual(camera_configs["front"].fourcc, "MJPG")
        self.assertEqual(camera_configs["front"].backend, Cv2Backends.V4L2)
        self.assertEqual(camera_configs["top"].backend, Cv2Backends.ANY)

    def test_missing_required_field_names_the_path(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Missing required config field: cameras.front.device"
        ):
            parse_hardware_config(
                {
                    "ports": {"follower": "/dev/ttyACM0"},
                    "cameras": {"front": {"width": 640, "height": 480, "fps": 30}},
                }
            )

    def test_mismatched_camera_fps_fails_fast(self) -> None:
        config = parse_hardware_config(
            {
                "ports": {"follower": "/dev/ttyACM0"},
                "cameras": {
                    "front": {"device": 0, "width": 640, "height": 480, "fps": 30},
                    "top": {"device": 1, "width": 640, "height": 480, "fps": 15},
                },
            }
        )

        with self.assertRaisesRegex(
            ValueError, "must share one fps value"
        ):
            config.require_shared_fps(("front", "top"))


if __name__ == "__main__":
    unittest.main()
