from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PRESET_PATHS = {
    "video": (
        "/dev/v4l/by-id",
        "/dev/v4l/by-path",
        "/dev/video*",
    ),
    "serial": (
        "/dev/serial/by-id",
        "/dev/serial/by-path",
        "/dev/ttyACM*",
        "/dev/ttyUSB*",
    ),
}


@dataclass(frozen=True)
class Entry:
    path: str
    target: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Watch device paths and print what changes when you disconnect or "
            "reconnect hardware. Useful for mapping stable IDs to physical devices."
        )
    )
    parser.add_argument(
        "--kind",
        choices=sorted(PRESET_PATHS),
        default="video",
        help="Preset device family to watch.",
    )
    parser.add_argument(
        "--path",
        dest="paths",
        action="append",
        default=[],
        help=(
            "Additional directory or glob to watch. Can be passed multiple times. "
            "Examples: /dev/input/by-id, /dev/hidraw*"
        ),
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Print one snapshot and exit.",
    )
    return parser.parse_args()


def expand_watch_specs(kind: str, extra_specs: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered_specs: list[str] = []
    for spec in (*PRESET_PATHS[kind], *extra_specs):
        if spec not in seen:
            seen.add(spec)
            ordered_specs.append(spec)
    return ordered_specs


def collect_entries(specs: list[str]) -> dict[str, Entry]:
    entries: dict[str, Entry] = {}
    for spec in specs:
        for path in expand_spec(spec):
            resolved = str(path.resolve()) if path.is_symlink() else None
            entries[str(path)] = Entry(path=str(path), target=resolved)
    return entries


def expand_spec(spec: str) -> list[Path]:
    path = Path(spec)
    if "*" in spec or "?" in spec or "[" in spec:
        return sorted(path.parent.glob(path.name))
    if path.is_dir():
        return sorted(path.iterdir())
    if path.exists() or path.is_symlink():
        return [path]
    return []


def print_snapshot(title: str, entries: dict[str, Entry]) -> None:
    print(title)
    if not entries:
        print("  (no matches)")
        return

    for entry_path in sorted(entries):
        entry = entries[entry_path]
        if entry.target is None:
            print(f"  {entry.path}")
        else:
            print(f"  {entry.path} -> {entry.target}")


def print_changes(previous: dict[str, Entry], current: dict[str, Entry]) -> bool:
    removed = sorted(set(previous) - set(current))
    added = sorted(set(current) - set(previous))

    if not removed and not added:
        return False

    print()
    print(f"Change detected at {time.strftime('%H:%M:%S')}")
    if removed:
        print("Removed:")
        for entry_path in removed:
            entry = previous[entry_path]
            if entry.target is None:
                print(f"  {entry.path}")
            else:
                print(f"  {entry.path} -> {entry.target}")
    if added:
        print("Added:")
        for entry_path in added:
            entry = current[entry_path]
            if entry.target is None:
                print(f"  {entry.path}")
            else:
                print(f"  {entry.path} -> {entry.target}")
    return True


def main() -> int:
    args = parse_args()
    specs = expand_watch_specs(args.kind, args.paths)

    print("Watching specs:")
    for spec in specs:
        print(f"- {spec}")

    baseline = collect_entries(specs)
    print()
    print_snapshot("Current devices:", baseline)

    if args.once:
        return 0

    print()
    print("Disconnect or reconnect a device. Press Ctrl+C to stop.")

    previous = baseline
    try:
        while True:
            time.sleep(args.interval)
            current = collect_entries(specs)
            if print_changes(previous, current):
                print()
                print_snapshot("Current devices:", current)
            previous = current
    except KeyboardInterrupt:
        print()
        print("Stopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
