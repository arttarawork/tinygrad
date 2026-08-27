# Ampere-over-Thunderbolt working fork (arttarawork/tinygrad, branch `memory`)

This is Artur's fork of tinygrad for making the NVIDIA/TinyGPU eGPU path fast for local LLMs
and pooling the MacBook (Metal) with an RTX 3090. Before doing any work, read in order:

1. `HANDOFF_2026-08-26.md` — the complete handoff after the 2026-08-26 dock incidents; then
   `TASKS.md` — **start with the "RESUME HERE" section** (state of play, where the artifacts
   live, dock ops quickstart, next-task recommendation), then the task list + Status log.
2. `NV_LLM_DESIGN.md` — the design doc; **§1.5 has the post-dock results and supersedes the
   pre-dock framing** in §1-§2.
3. `memory.md` — context, decision history, external research, supplementary repo findings.
4. For any work touching the dock: `TD3_POOLING_NOTES.md` §0 and `BENCH_NOTES.md` on branch
   `task/TD.3-pooling` (worktree `/Users/artur/Documents/tinygrad-dock`) — lane mechanics and
   all measured data live there, not on this branch.

Docs were baselined at upstream `af2a43c85` (2026-08-18); the 2026-08-26 panic analyses (`T4.36_DART_PANIC.md`,
HANDOFF §2) cite `integration/phase1b` (`b37c792c6`). Re-verify file:line refs after rebases.

## Machine roles
- **MacBook M3 Pro 36 GB** (`ENV:MAC`): Metal perf work, mock-NV, METAL+CPU pooling rehearsal.
  The eGPU dock (AOOSTAR AG02) + RTX 3090 went live 2026-08-24 (TD.1→TD.3 done, truth table measured) —
  but **ALL DOCK WORK IS HARD-STOPPED since 2026-08-26 after two host kernel panics** (HANDOFF §2/§5):
  no `DEV=NV*`, no NV-opening pytest, no `pkill` of a TinyGPU server, until Artur's §5.1 decision.
- **Bazzite box, RX 9070 XT** (`ENV:AMD`): descoped 2026-08-18 (AMD is not a target; shared-HCQ
  validation goes via mock-NV + rented 3090 instead — see TASKS.md T0.2/T2.1). If ever revived:
  KFD iface only — never the AM/PCI driver path there; it unbinds amdgpu and kills the display.

## Conventions
- Branch `task/T<id>-<slug>` off fork `master` = **`7ca254099`** (the PR #6 merge: upstream sync #4 + the COMPLETE hardware-verified remediation T4.37/40-1/40-2/40-3/40-4; fork line-cap 27000). `integration/phase1b`
  RETIRED; dock trees = master or a descendant — plus `task/T4.40c-halt-verify` once hardware-verified. There is no local `master`, and **`origin/*` tracking refs are STALE by construction** (SSH `origin`
  is interactive-only; agents fetch/push via the explicit HTTPS URL `https://github.com/arttarawork/tinygrad.git`,
  which never updates `origin/*`) — **never branch off `origin/master`** (it sits at the 08-21 sync `b37d80fc9`);
  verify push state with `gh api repos/arttarawork/tinygrad/branches`. Older baselines (`af2a43c85`,
  `457e1a915`, `b37d80fc9`) apply only to their era's branches. Remotes: `origin` = arttarawork/tinygrad (the
  fork), `upstream` = tinygrad/tinygrad. Rebase docs branch weekly and **push `memory` + unmerged evidence
  branches after each session** (backup; see TASKS.md conventions).
- Python: there is **no bare `python`** on this Mac and Homebrew `python3` (3.14) has no test deps.
  Use the repo venv: `DEV=CPU PYTHONPATH=. .venv/bin/python -m pytest <area> -x -q` (serial; `-n12` only under
  `DEV=CPU` — with the METAL default `-n` spawns one real TinyGPU server per worker, T4.38; `DEV=CPU` gates never
  open NV except `test/device/test_hcq.py`, which opens `Device["NV"]` unconditionally);
  typecheck `.venv/bin/python -m mypy tinygrad/`; lint `.venv/bin/python -m ruff check .`
  (from a worktree, the venv is at `/Users/artur/Documents/tinygrad/.venv` — `PYTHONPATH=.` makes
  the worktree's tinygrad win over anything installed).
- Perf claims need before/after tok/s from the T0.3 harness on named hardware. Upstream PRs:
  one small lever each, hand-verified — upstream has reverted AI-generated slop before.
- Don't remove the deliberate `.contiguous()` in the MoE expert path (`tinygrad/llm/model.py:27,129`).
- Subagents: when work breaks into discrete, well-defined ("Sonnet-proof") subtasks, hand them to
  **Sonnet 5 at max effort** (`model: "sonnet"`, request max reasoning in the prompt) — one tight
  objective, explicit done-when + STOP conditions. Don't spawn inherited-Fable agents for these.
