import unittest

import motor_usb


class MotorUsbHelpersTests(unittest.TestCase):
    def test_parse_hex_bytes_spaced_tokens(self) -> None:
        self.assertEqual(motor_usb.parse_hex_bytes("FF 0x01 2"), b"\xFF\x01\x02")

    def test_parse_hex_bytes_compact(self) -> None:
        self.assertEqual(motor_usb.parse_hex_bytes("A10B"), b"\xA1\x0B")

    def test_parse_hex_bytes_rejects_invalid(self) -> None:
        with self.assertRaises(ValueError):
            motor_usb.parse_hex_bytes("GG")

    def test_format_frame_prints_hex_and_ascii(self) -> None:
        self.assertEqual(motor_usb.format_frame(b"A\x00"), "41 00    A.")

    def test_build_text_payload_newline_toggle(self) -> None:
        self.assertEqual(motor_usb.build_text_payload("abc", newline=True), b"abc\n")
        self.assertEqual(motor_usb.build_text_payload("abc", newline=False), b"abc")


if __name__ == "__main__":
    unittest.main()
