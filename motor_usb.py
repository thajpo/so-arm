"""Compatibility entrypoint for the SO-101 USB motor utility."""

from hardware_tests.motor_usb import *  # noqa: F401,F403
from hardware_tests.motor_usb import main


if __name__ == "__main__":
    raise SystemExit(main())
