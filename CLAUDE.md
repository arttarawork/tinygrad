# Ampere-over-Thunderbolt working fork (arttarawork/tinygrad, branch `memory`)

This is Artur's fork of tinygrad for making the NVIDIA/TinyGPU eGPU path fast for local LLMs
and pooling the MacBook (Metal) with an RTX 3090. Before doing any work, read in order:

1. **`HANDOFF_2026-09-01.md` — THE entry point** (08-31 = 3.8-chain history, 08-27 = TD.5 history) (fault era closed, the two-bug WY record, the board,
   standing rules/grants, verification gotchas, open board); then `TASKS.md` — the "RESUME HERE"
   section, then the task list + Status log. (`HANDOFF_2026-08-26.md` is the panic-era history.)
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
  The eGPU dock (AOOSTAR AG02) + RTX 3090 is **LIVE with autonomous `DEV=NV` granted** (the 08-26
  panics are fully remediated + hardware-verified; M3 achieved: Q8_0 pooled 31.1 tok/s). Standing
  protocol in HANDOFF_2026-08-27 §4 — above all: **never kill an NV process or TinyGPU server**.
  **TD.5 (pooled model selectable in Hermes) is WIRED as of 2026-08-27 PM** — ritual in `~/CLAUDE.md` "Pooled model",
  state + the next levers in HANDOFF_2026-08-27 §2/§3. Serving tree = `tinygrad-td5` (on fork master).
- **Bazzite box, RX 9070 XT** (`ENV:AMD`): descoped 2026-08-18 (AMD is not a target; shared-HCQ
  validation goes via mock-NV + rented 3090 instead — see TASKS.md T0.2/T2.1). If ever revived:
  KFD iface only — never the AM/PCI driver path there; it unbinds amdgpu and kills the display.

## Conventions
- Branch `task/T<id>-<slug>` off fork `master` = **`2d38cada5`** (the PR #26 merge, 2026-09-01 — #25 WPR ceiling + #26 forensics in: the FULL chain is on master — UD quants, scan harness+split, MTP load, greedy+sampled speculative decode, --mtp serve, head-group scan split, WY scan+decode gate, draft-pos Variable, state cache, chunk-64 coverage, FFN TP, CI test-cost fix. No open PRs; WY/state-cache/MTP ship OFF by default pending T4.73/T4.74.)
  `integration/phase1b`
  RETIRED; dock trees = master or a descendant — plus `task/T4.40c-halt-verify` once hardware-verified. There is no local `master`, and **`origin/*` tracking refs are STALE by construction** (SSH `origin`
  is interactive-only; agents fetch/push via the explicit HTTPS URL `https://github.com/arttarawork/tinygrad.git`,
  which never updates `origin/*`) — **never branch off `origin/master`** (it sits at the 08-21 sync `b37d80fc9`);
  verify push state with `gh api repos/arttarawork/tinygrad/branches`. Older baselines (`af2a43c85`,
  `457e1a915`, `b37d80fc9`) apply only to their era's branches. Remotes: `origin` = arttarawork/tinygrad (the
  fork), `upstream` = tinygrad/tinygrad. Rebase docs branch weekly and **push `memory` + unmerged evidence
  branches after each session** (backup; see TASKS.md conventions).
- Python: there is **no bare `python`** on this Mac and Homebrew `python3` (3.14) has no test deps.
  Use the repo venv: `CHECK_OOB=1 DEV=CPU PYTHONPATH=. .venv/bin/python -m pytest <area> -x -q` (CI sets `CHECK_OOB=1`; serial; `-n12` only under
  `DEV=CPU` — with the METAL default `-n` spawns one real TinyGPU server per worker, T4.38; `DEV=CPU` gates never
  open NV except `test/device/test_hcq.py`, which opens `Device["NV"]` unconditionally);
  typecheck `.venv/bin/python -m mypy tinygrad/`; lint `.venv/bin/python -m ruff check .` AND the CI whitespace lint
  `.venv/bin/python -m pylint --disable=all -e W0311 -e C0303 --jobs=0 --indent-string='  ' --recursive=y .` (2-space indent — 4-space
  scripts in `extra/` fail CI's Linters job; pylint must be installed in the venv — it wasn't until 2026-09-03)
  (from a worktree, the venv is at `/Users/artur/Documents/tinygrad/.venv` — `PYTHONPATH=.` makes
  the worktree's tinygrad win over anything installed).
- Perf claims need before/after tok/s from the T0.3 harness on named hardware. Upstream PRs:
  one small lever each, hand-verified — upstream has reverted AI-generated slop before.
- Don't remove the deliberate `.contiguous()` in the MoE expert path (`tinygrad/llm/model.py:27,129`).
- Subagents: when work breaks into discrete, well-defined ("Sonnet-proof") subtasks, hand them to
  **Sonnet 5 at max effort** (`model: "sonnet"`, request max reasoning in the prompt) — one tight
  objective, explicit done-when + STOP conditions. Don't spawn inherited-Fable agents for these.
