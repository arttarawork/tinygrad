# Ampere-over-Thunderbolt working fork (arttarawork/tinygrad, branch `memory`)

This is Artur's fork of tinygrad for making the NVIDIA/TinyGPU eGPU path fast for local LLMs
and pooling the MacBook (Metal) with an RTX 3090. Before doing any work, read in order:

1. `TASKS.md` — the task list with dependencies and environments; pick tasks from there.
2. `NV_LLM_DESIGN.md` — the design doc (goals, current-state findings with file:line refs, workstreams).
3. `memory.md` — context, decision history, external research, supplementary repo findings.

Docs are baselined at upstream `af2a43c85` (2026-08-18); re-verify file:line refs after rebases.

## Machine roles
- **MacBook M3 Pro 36 GB** (`ENV:MAC`): Metal perf work, mock-NV, METAL+CPU pooling rehearsal.
  The eGPU dock (AOOSTAR AG02) + RTX 3090 arrived 2026-08-23 — `DOCK`-tagged tasks are
  unblocked; TD.1 bring-up is the gate (procedure + preflight in TASKS.md TD.1).
- **Bazzite box, RX 9070 XT** (`ENV:AMD`): descoped 2026-08-18 (AMD is not a target; shared-HCQ
  validation goes via mock-NV + rented 3090 instead — see TASKS.md T0.2/T2.1). If ever revived:
  KFD iface only — never the AM/PCI driver path there; it unbinds amdgpu and kills the display.

## Conventions
- Branch `task/T<id>-<slug>` off fork `master` (`457e1a915` since 2026-08-21 = the PR #1 merge:
  all Phase 0 work + upstream `b8cc74ecf`; there is no local `master` — use `origin/master`).
  The old baseline `af2a43c85` applies only to the original Phase 0 branches. Remotes: `origin` =
  arttarawork/tinygrad (the fork), `upstream` = tinygrad/tinygrad. Rebase docs branch weekly and
  **push `memory` + unmerged evidence branches after each session** (backup; see TASKS.md conventions).
- Python: there is **no bare `python`** on this Mac and Homebrew `python3` (3.14) has no test deps.
  Use the repo venv: `PYTHONPATH=. .venv/bin/python -m pytest <area> -x -q -n12`;
  typecheck `.venv/bin/python -m mypy tinygrad/`; lint `.venv/bin/python -m ruff check .`
  (from a worktree, the venv is at `/Users/artur/Documents/tinygrad/.venv` — `PYTHONPATH=.` makes
  the worktree's tinygrad win over anything installed).
- Perf claims need before/after tok/s from the T0.3 harness on named hardware. Upstream PRs:
  one small lever each, hand-verified — upstream has reverted AI-generated slop before.
- Don't remove the deliberate `.contiguous()` in the MoE expert path (`tinygrad/llm/model.py:27,129`).
