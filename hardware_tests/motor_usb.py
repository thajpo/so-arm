#!/usr/bin/env python3
"""Simple USB serial utility to inspect and control a motor controller."""

from __future__ import annotations

import argparse
import string
import sys
import time
from dataclasses import dataclass
from typing import Sequence

try:
    import serial
    from serial.tools import list_ports
except ModuleNotFoundError:
    serial = None
    list_ports = None

try:
    from serial import Serial as SerialClass
except Exception:
    SerialClass = None
    try:
        from serial.serialposix import Serial as SerialClass  # macOS/Linux fallback
    except Exception:
        try:
            from serial.serialwin32 import Serial as SerialClass  # Windows fallback
        except Exception:
            SerialClass = None

DEFAULT_BAUD = 1_000_000


@dataclass(frozen=True)
class SerialPortInfo:
    device: str
    description: str
    hwid: str


def list_serial_ports() -> list[SerialPortInfo]:
    if list_ports is None:
        raise RuntimeError("pyserial is required. Install dependencies with: uv sync")
    ports: list[SerialPortInfo] = []
    for port in list_ports.comports():
        ports.append(
            SerialPortInfo(
                device=port.device,
                description=port.description or "",
                hwid=port.hwid or "",
            )
        )
    return ports


def guess_default_port(ports: Sequence[SerialPortInfo]) -> str | None:
    if not ports:
        return None
    for port in ports:
        joined = f"{port.device} {port.description}".lower()
        if "usb" in joined or "modem" in joined:
            return port.device
    return ports[0].device


def resolve_port(requested: str | None) -> str:
    ports = list_serial_ports()
    if requested:
        known = {port.device for port in ports}
        if requested not in known:
            raise RuntimeError(
                f"Port '{requested}' not found. Run 'python motor_usb.py list' first."
            )
        return requested

    guessed = guess_default_port(ports)
    if guessed is None:
        raise RuntimeError("No serial ports detected.")
    return guessed


def open_serial(port: str, baud: int, timeout: float):
    if serial is None:
        raise RuntimeError("pyserial is required. Install dependencies with: uv sync")
    if SerialClass is None:
        raise RuntimeError(
            "Could not load pyserial Serial backend for this OS. "
            "Re-run 'uv sync' and verify pyserial install."
        )
    return SerialClass(port=port, baudrate=baud, timeout=timeout, write_timeout=1.0)


def parse_hex_bytes(value: str) -> bytes:
    cleaned = value.strip().replace(",", " ")
    if not cleaned:
        raise ValueError("Hex payload cannot be empty.")

    if " " not in cleaned and cleaned.startswith("0x"):
        cleaned = cleaned[2:]

    if " " not in cleaned and len(cleaned) > 2:
        if len(cleaned) % 2 != 0:
            raise ValueError("Compact hex payload must have an even number of digits.")
        if any(ch not in string.hexdigits for ch in cleaned):
            raise ValueError(f"Invalid hex payload: {value!r}")
        return bytes.fromhex(cleaned)

    parsed: list[int] = []
    for raw_token in cleaned.split():
        token = raw_token[2:] if raw_token.lower().startswith("0x") else raw_token
        if not token or len(token) > 2:
            raise ValueError(f"Invalid hex token: {raw_token!r}")
        if any(ch not in string.hexdigits for ch in token):
            raise ValueError(f"Invalid hex token: {raw_token!r}")
        parsed.append(int(token, 16))
    return bytes(parsed)


def format_frame(data: bytes) -> str:
    hex_view = " ".join(f"{byte:02X}" for byte in data)
    ascii_view = "".join(chr(byte) if 32 <= byte <= 126 else "." for byte in data)
    return f"{hex_view}    {ascii_view}"


def build_text_payload(text: str, newline: bool) -> bytes:
    payload = text.encode("utf-8")
    if newline:
        payload += b"\n"
    return payload


def read_for_window(serial_port, seconds: float) -> bytes:
    deadline = time.monotonic() + max(seconds, 0.0)
    chunks: list[bytes] = []
    while time.monotonic() < deadline:
        waiting = int(getattr(serial_port, "in_waiting", 0))
        block = serial_port.read(waiting if waiting > 0 else 1)
        if block:
            chunks.append(block)
        else:
            time.sleep(0.01)
    return b"".join(chunks)


def cmd_list(_: argparse.Namespace) -> int:
    ports = list_serial_ports()
    if not ports:
        print("No serial devices detected.")
        return 0

    guessed = guess_default_port(ports)
    print("Detected serial devices:")
    for port in ports:
        marker = " (default)" if guessed == port.device else ""
        print(f"- {port.device}{marker}")
        if port.description:
            print(f"  description: {port.description}")
        if port.hwid:
            print(f"  hwid: {port.hwid}")
    return 0


def cmd_monitor(args: argparse.Namespace) -> int:
    port = resolve_port(args.port)
    print(
        f"Monitoring {port} at {args.baud} baud for {args.seconds:.1f}s "
        "(Ctrl+C to stop)..."
    )
    started = time.monotonic()
    deadline = started + args.seconds
    with open_serial(port=port, baud=args.baud, timeout=0.2) as serial_port:
        while time.monotonic() < deadline:
            waiting = int(getattr(serial_port, "in_waiting", 0))
            data = serial_port.read(waiting if waiting > 0 else 1)
            if not data:
                continue
            elapsed = time.monotonic() - started
            print(f"[{elapsed:7.3f}s] RX {format_frame(data)}")
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    port = resolve_port(args.port)
    if args.text is not None:
        payload = build_text_payload(args.text, newline=not args.no_newline)
    else:
        payload = parse_hex_bytes(args.hex_payload)

    with open_serial(port=port, baud=args.baud, timeout=0.05) as serial_port:
        if hasattr(serial_port, "reset_input_buffer"):
            serial_port.reset_input_buffer()
        serial_port.write(payload)
        serial_port.flush()
        print(f"TX {format_frame(payload)}")
        response = read_for_window(serial_port, seconds=args.read_seconds)
        if response:
            print(f"RX {format_frame(response)}")
        else:
            print("RX <no bytes>")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and send commands to a USB motor controller over serial "
            "(defaults tuned for STS3215 bus servos)."
        )
    )
    subparsers = parser.add_subparsers(dest="command")

    list_parser = subparsers.add_parser("list", help="List detected serial devices")
    list_parser.set_defaults(func=cmd_list)

    monitor_parser = subparsers.add_parser(
        "monitor", help="Print incoming bytes from the serial device"
    )
    monitor_parser.add_argument(
        "--port", help="Serial port path (auto-detect if omitted)"
    )
    monitor_parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate (default: {DEFAULT_BAUD})",
    )
    monitor_parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Monitoring duration in seconds (default: 10.0)",
    )
    monitor_parser.set_defaults(func=cmd_monitor)

    send_parser = subparsers.add_parser(
        "send", help="Send one text or hex command and print the response"
    )
    send_parser.add_argument("--port", help="Serial port path (auto-detect if omitted)")
    send_parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD,
        help=f"Baud rate (default: {DEFAULT_BAUD})",
    )
    send_group = send_parser.add_mutually_exclusive_group(required=True)
    send_group.add_argument(
        "--text",
        help="UTF-8 text payload to send (newline is appended unless --no-newline is set)",
    )
    send_group.add_argument(
        "--hex-payload",
        help='Hex payload to send, e.g. "FF 01 00" or "FF0100"',
    )
    send_parser.add_argument(
        "--no-newline",
        action="store_true",
        help="Do not append newline to --text payload",
    )
    send_parser.add_argument(
        "--read-seconds",
        type=float,
        default=0.75,
        help="How long to wait for response bytes after send (default: 0.75)",
    )
    send_parser.set_defaults(func=cmd_send)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0

    try:
        return int(args.func(args))
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
