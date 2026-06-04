from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from camera_wall import compute_grid, discover_index0_targets, shorten_by_path_label


class CameraWallTests(unittest.TestCase):
    def test_compute_grid_prefers_compact_layout(self) -> None:
        self.assertEqual(compute_grid(1), (1, 1))
        self.assertEqual(compute_grid(2), (1, 2))
        self.assertEqual(compute_grid(3), (2, 2))
        self.assertEqual(compute_grid(5), (2, 3))

    def test_compute_grid_rejects_zero(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            compute_grid(0)

    def test_shorten_by_path_label_keeps_usb_port_identity(self) -> None:
        self.assertEqual(
            shorten_by_path_label("pci-0000:80:14.0-usb-0:11.2.1:1.0-video-index0"),
            "port 0:11.2.1:1.0",
        )

    def test_discover_index0_targets_filters_duplicate_nodes(self) -> None:
        with self.subTest("build fake /dev/v4l/by-path tree"):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp_dir:
                root = Path(tmp_dir)
                devices = root / "devices"
                devices.mkdir()
                (devices / "video2").touch()
                (devices / "video3").touch()
                (devices / "video4").touch()
                links = root / "by-path"
                links.mkdir()
                (links / "pci-x-usb-0:11.2.1:1.0-video-index0").symlink_to(devices / "video2")
                (links / "pci-x-usb-0:11.2.1:1.0-video-index1").symlink_to(devices / "video3")
                (links / "pci-x-usb-0:11.2.2:1.0-video-index0").symlink_to(devices / "video4")

                targets = discover_index0_targets(links)

        self.assertEqual([target.device for target in targets], [str(devices / "video2"), str(devices / "video4")])


if __name__ == "__main__":
    unittest.main()
