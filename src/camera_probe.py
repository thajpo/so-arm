import time

import cv2

REQUESTED_WIDTH = 640
REQUESTED_HEIGHT = 480
REQUESTED_FPS = 30
CAMERA_INDICES = [0, 1]
WARMUP_SECONDS = 1.0
TEST_SECONDS = 20.0
LOG_EVERY_N_FRAMES = 30


def open_camera(index: int) -> cv2.VideoCapture | None:
    print(f"\n=== Opening camera {index} ===")
    cap = cv2.VideoCapture(index)
    print("opened:", cap.isOpened())
    if not cap.isOpened():
        return None

    try:
        print("backend:", cap.getBackendName())
    except Exception as exc:
        print("backend: <unavailable>", repr(exc))

    print("set width:", cap.set(cv2.CAP_PROP_FRAME_WIDTH, REQUESTED_WIDTH))
    print("set height:", cap.set(cv2.CAP_PROP_FRAME_HEIGHT, REQUESTED_HEIGHT))
    print("set fps:", cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS))

    time.sleep(0.5)

    print("reported width :", cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    print("reported height:", cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("reported fps   :", cap.get(cv2.CAP_PROP_FPS))

    return cap


def read_frame(cap: cv2.VideoCapture, index: int) -> tuple[bool, object]:
    ok, frame = cap.read()
    if not ok:
        print(f"[camera {index}] read failed")
    return ok, frame


def main() -> None:
    print(
        f"Concurrent probe for cameras {CAMERA_INDICES} at "
        f"{REQUESTED_WIDTH}x{REQUESTED_HEIGHT} @ {REQUESTED_FPS} FPS"
    )

    caps: dict[int, cv2.VideoCapture] = {}
    stats = {
        index: {
            "reads": 0,
            "failures": 0,
            "first_shape": None,
            "last_ok_at": None,
        }
        for index in CAMERA_INDICES
    }

    try:
        for index in CAMERA_INDICES:
            cap = open_camera(index)
            if cap is None:
                print(f"Aborting: camera {index} did not open.")
                return
            caps[index] = cap

        print(f"\nWarming up both cameras for {WARMUP_SECONDS:.1f}s...")
        warmup_deadline = time.time() + WARMUP_SECONDS
        while time.time() < warmup_deadline:
            for index, cap in caps.items():
                ok, frame = read_frame(cap, index)
                if ok:
                    stats[index]["last_ok_at"] = time.time()
                    if stats[index]["first_shape"] is None and frame is not None:
                        stats[index]["first_shape"] = frame.shape

        print(f"\nStarting concurrent stress test for {TEST_SECONDS:.1f}s...")
        start = time.time()
        deadline = start + TEST_SECONDS

        while time.time() < deadline:
            for index, cap in caps.items():
                ok, frame = read_frame(cap, index)
                stats[index]["reads"] += 1

                if ok:
                    stats[index]["last_ok_at"] = time.time()
                    if stats[index]["first_shape"] is None and frame is not None:
                        stats[index]["first_shape"] = frame.shape
                else:
                    stats[index]["failures"] += 1

                if stats[index]["reads"] % LOG_EVERY_N_FRAMES == 0:
                    elapsed = time.time() - start
                    approx_fps = stats[index]["reads"] / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[camera {index}] reads={stats[index]['reads']} "
                        f"failures={stats[index]['failures']} "
                        f"approx_fps={approx_fps:.2f} "
                        f"shape={stats[index]['first_shape']}"
                    )

        print("\n=== Summary ===")
        total_elapsed = time.time() - start
        for index in CAMERA_INDICES:
            reads = stats[index]["reads"]
            failures = stats[index]["failures"]
            approx_fps = reads / total_elapsed if total_elapsed > 0 else 0.0
            print(f"\nCamera {index}")
            print("first shape :", stats[index]["first_shape"])
            print("total reads :", reads)
            print("failures    :", failures)
            print("approx fps  :", f"{approx_fps:.2f}")
            print("last ok at  :", stats[index]["last_ok_at"])

    finally:
        for index, cap in caps.items():
            cap.release()
            print(f"released camera {index}")


if __name__ == "__main__":
    main()
