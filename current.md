# current.md

## Institutional Knowledge
- Hardware attachment details differ across machines. macOS used `/dev/cu.*` device names; Linux uses `/dev/ttyACM*` or `/dev/ttyUSB*`, and camera index ordering may also change.
- Current runtime entrypoints hardcode device assumptions in source, notably [src/run_policy.py](/home/j/Projects/so-arm/src/run_policy.py#L86) and [src/record.py](/home/j/Projects/so-arm/src/record.py#L39).
- Repo workflow is spec-first: discuss, approve a concrete spec here, then promote one ready item to a GitHub issue before implementation.

## Beliefs
- Machine-specific hardware settings should not live in application code.
- The first version should be explicit and inspectable, not auto-detect-heavy.
- We should minimize new dependencies unless the ergonomics win is meaningful.

## Brainstormed
- none currently

## Specd

### Hardware Config Externalization
- Status: ready
- User intent:
  Make hardware setup machine-agnostic so moving between macOS and Linux only requires editing a config file, not Python source.
- Problem:
  `src/run_policy.py` and `src/record.py` embed camera indices and serial ports directly in code. That makes the repo brittle across machines, because Linux and macOS enumerate USB cameras and serial devices differently. The immediate failure mode is port/index churn when moving between your Mac and Linux workstation.
- Decision:
  Introduce one hardware configuration file loaded at runtime, with a small shared loader used by both scripts. Use TOML for v1 so parsing stays in the Python standard library via `tomllib` and we avoid a new dependency.
- Proposed scope:
  Add a checked-in example config plus an ignored local override file.
  Move follower port, leader port, and named camera definitions out of source and into config.
  Share config loading/validation logic across `src/run_policy.py` and `src/record.py`.
  Keep the first version explicit: no USB auto-detection beyond clear validation errors.
  Update docs with the new setup and verification commands.
- Acceptance criteria:
  `src/run_policy.py` no longer hardcodes serial ports or camera indices.
  `src/record.py` no longer hardcodes serial ports or camera indices.
  A user can configure Linux and macOS hardware without editing Python source.
  Missing or malformed config fails fast with a clear error naming the missing field.
  At least one automated test covers config parsing/validation.
- Testing plan:
  Add unit tests for config load/validation behavior, including missing required keys and a valid sample config.
- Likely files:
  `current.md`
  `src/run_policy.py`
  `src/record.py`
  `src/hardware_config.py`
  `tests/test_hardware_config.py`
  `README.md`
- File touch scope:
  `current.md`
  `src/run_policy.py`
  `src/record.py`
  `src/hardware_config.py`
  `tests/test_hardware_config.py`
  `README.md`
- Approval:
  Approved by user in chat on 2026-03-21 after TOML vs YAML review; implementation requested with "ok, do it."
