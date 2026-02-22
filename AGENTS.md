# AGENTS

## Project Intent
Build a robust sim-to-real learning stack for SO-101 manipulation, starting with a single-arm v0 baseline and expanding to bimanual tasks.

## User Goals
- Learn deeply by discussing tradeoffs before implementation.
- Improve systems-engineering discipline: clear specs, explicit acceptance criteria, and test-backed feature delivery.
- Maintain fast iteration loops with reproducible experiments and minimal complexity until needed.

## Collaboration Workflow (Required)
1. Discuss ideas and narrow to what matters.
2. Write/approve a concrete feature spec in `current.md` (`Specd` section).
3. Convert one `Specd: Ready` item into one GitHub issue with explicit file-touch scope.
4. Implement the feature from that issue in one isolated branch/worktree.
5. Add/adjust tests so the feature has at least one passing test proving behavior.
6. Share concise implementation notes + test results.
7. User reviews/merges, then repeat.

## Implementation Principles
- Prefer simple baselines first (clear failures > complex stacks).
- Keep configs explicit and reproducible (seeded runs, deterministic eval where possible).
- Avoid premature scale work (e.g., multi-GPU) unless profiling shows bottlenecks.
- Preserve existing code unless spec requires change.
- Every merged feature should be testable from CLI.

## Definition of Done (Per Feature)
A feature is done only when all are true:
- Spec section exists in `current.md` and is marked `done`/pruned per workflow.
- Acceptance criteria are satisfied.
- At least one associated automated test passes.
- Any relevant docs/commands are updated.

## Spec Status Labels
- `Backlog`: discussed, not started.
- `Ready`: specified and approved for implementation.
- `In Progress`: being implemented.
- `Implemented`: code merged and tests passing.
- `Deferred`: intentionally paused.

## Communication Style
- Prioritize direct, technical clarity.
- Call out assumptions and unknowns explicitly.
- Prefer proposing concrete defaults when user is unsure.

## Copy-Paste Workflow Template (For Other Projects/Sessions)
status: done

Use this block directly in a new repo/session:

```md
# Engineering Workflow (Spec-First, Terminal-First)

## Primary Control Surface
- Use one markdown file as the canonical planning/discussion log.
- Keep per-topic sections with:
  - Context
  - Open Questions
  - Decision
  - Action Items
  - Status: open|ready|in_progress|review|done
- Append updates under active section, then prune stale text after decisions lock.

## User Touch Points
1. Approve feature spec markdown.
2. Review and approve PR.

## Required Delivery Gates
1. Spec approved (`ready`) before coding.
2. Isolated implementation branch/worktree per feature.
3. Test-first:
   - add/identify failing test,
   - implement fix/feature,
   - verify pass.
4. Run regression tests for touched surfaces.
5. PR summary must include:
   - scope/non-goals,
   - changed files,
   - fail->pass evidence,
   - risk notes.
6. Review loop until accepted.

## Guardrails
- No implementation without clear spec.
- No disabling/skipping existing tests to make feature pass.
- Prefer explicit modes over ambiguous defaults.
- Prefer fail-fast contracts over backward-compat shims unless explicitly required.
```
