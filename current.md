# current

Last updated: 2026-02-14
Canonical planning/discussion file for this repo (lean-flow).

## Institutional Knowledge
- Project intent: build a robust sim-to-real SO-101 stack, starting with single-arm v0 and expanding to bimanual later.
- Workflow contract: spec-first, terminal-first, test-first.
- Workflow policy (issue/PR contract):
  - `Specd` promotion to `ready` requires explicit user intent/approval evidence.
  - A `ready` Specd item must be converted to exactly one GitHub issue before implementation.
  - Each issue must declare file-touch scope before coding starts.
  - Execution mapping is strict: one issue -> one branch -> one worktree -> one PR.
- Specd lifecycle contract:
  - `ready` -> `in_progress` -> `review` (PR open/under review)
  - prune Specd item only after PR merge
  - no merged-note retained in `current.md` after prune
- Delivery gates:
  - Spec must be `ready` before coding.
  - Implement in isolated unit of work.
  - Show fail->pass test evidence.
  - Run regression tests for touched surfaces.
  - Include PR summary with scope/non-goals, changed files, fail->pass evidence, risk notes.
- Guardrails:
  - No implementation from loose brainstorm notes.
  - No disabling/skipping tests to force pass.
  - Prefer explicit modes and fail-fast contracts.
- Current repo truth:
  - Single-arm scaffold exists: `single_arm_v0.py`, `models/so101/single_arm_task.xml`.
  - Bimanual scaffold exists but is deprioritized for now: `bimanual.py`, `models/so101/bimanual.xml`.

## Beliefs
- [2026-02-07] Single-arm v0 should be completed before new bimanual work.
  - Rationale: lower debug surface; faster sim2real iteration.
  - Evidence: prior decision log, now captured in this file's `Specd`/`Brainstormed` continuity.
- [2026-02-07] Baseline should stay simple before scale features (e.g., multi-GPU).
  - Rationale: iteration speed and observability matter more than throughput this early.
  - Evidence: AGENTS implementation principles.
- [2026-02-07] Viewer/inspection should land before PPO baseline.
  - Rationale: deterministic debugging loop reduces wasted training cycles.
  - Evidence: `Specd` S1 is ready while PPO remains brainstormed.

## Brainstormed
### B1. PPO Baseline for Single-Arm v0
- Status: open
- Notes:
  - single-process PPO first
  - explicit seed/config
  - checkpoint + eval command
  - minimal pass/fail gate

### B2. Bimanual Return Path
- Status: open
- Notes:
  - resume only after single-arm viewer + PPO baseline are done
  - port stable tooling from single-arm first

### Camera in Environment
- Status: open
- Notes:
  - idea: add camera support to env observations/render path
  - unresolved scope:
    - option A: single-arm env first (`single_arm_v0.py`)
    - option B: bimanual env first (`bimanual.py`)
  - default recommendation: implement camera path in single-arm first, then port to bimanual
  - decision needed before promotion to `Specd`: camera type(s), resolution, and whether viewer-only or policy observation path

## Specd
### S1. Single-Arm v0 Viewer + Rollout Inspection
- Status: ready
- behavior change:
  - Add a deterministic rollout inspection entrypoint for `single_arm_v0` with optional viewer.
  - Standardize per-episode diagnostics and aggregate summary output.
- files to touch:
  - `view_single_arm_v0.py` (new)
  - `README.md` (usage snippet update)
  - `tests/test_view_single_arm_v0.py` (new)
- fail-first tests:
  - CLI smoke test fails before script exists / passes after.
  - Determinism test: same seed/mode/steps yields identical diagnostics.
  - Output contract test: required keys present.
- non-goals:
  - No PPO training implementation in this spec.
  - No bimanual feature changes.
  - No advanced GUI overlays.
- risks:
  - Viewer behavior may differ by host/display availability.
  - Nondeterminism if RNG/state handling is inconsistent across reset paths.
- touch points (path + function/class/block):
  - `view_single_arm_v0.py`: CLI parser, rollout loop, diagnostics formatter, summary block.
  - `single_arm_v0.py`: env reset/step API consumption only (avoid behavior changes unless needed for determinism hooks).
  - `tests/test_view_single_arm_v0.py`: CLI smoke/determinism/output tests.
  - `README.md`: runnable command examples.
- line anchors (optional, reviewer convenience only):
  - `single_arm_v0.py` around `class SingleArmSO101Env`, `reset`, `step`.
- expected diff shape (add/modify/delete + rough LOC):
  - add: 2 files, ~180-260 LOC (`view_single_arm_v0.py`, tests)
  - modify: 1-2 files, ~10-40 LOC (`README.md`, possibly small env hook)
  - delete: none
- review checks:
  - `uv run view_single_arm_v0.py --help` shows:
    - `--mode {zero,random,scripted}`
    - `--episodes`
    - `--max-steps`
    - `--seed`
    - `--no-viewer`
  - `uv run view_single_arm_v0.py --no-viewer --mode random --episodes 2 --max-steps 50 --seed 7` prints:
    - 2 episode diagnostics with keys: `success,dropped,direction,episode_return,steps,seed`
    - final summary: `episodes,success_rate,drop_rate,mean_return`
  - automated tests pass.
