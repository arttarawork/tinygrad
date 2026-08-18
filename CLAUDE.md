# Ampere-over-Thunderbolt working fork (arttarawork/tinygrad, branch `memory`)

This is Artur's fork of tinygrad for making the NVIDIA/TinyGPU eGPU path fast for local LLMs
and pooling the MacBook (Metal) with an RTX 3090. Before doing any work, read in order:

1. `TASKS.md` — the task list with dependencies and environments; pick tasks from there.
2. `NV_LLM_DESIGN.md` — the design doc (goals, current-state findings with file:line refs, workstreams).
3. `memory.md` — context, decision history, external research, supplementary repo findings.

Docs are baselined at upstream `af2a43c85` (2026-08-18); re-verify file:line refs after rebases.

## Machine roles
- **MacBook M3 Pro 36 GB** (`ENV:MAC`): Metal perf work, mock-NV, METAL+CPU pooling rehearsal.
  Until the eGPU dock (AOOSTAR AG02) arrives, `DOCK`-tagged tasks are blocked.
- **Bazzite box, RX 9070 XT** (`ENV:AMD`): HCQ testbed via `DEV=AMD` (KFD iface only —
  never the AM/PCI driver path there; it unbinds amdgpu and kills the display).

## Conventions
- Branch `task/T<id>-<slug>` off `master`; keep `master` clean for upstream sync (remote `origin` =
  upstream tinygrad, remote `fork` = arttarawork/tinygrad). Rebase docs branch weekly.
- Tests: `PYTHONPATH=. python -m pytest <area> -x -q -n12`; typecheck `python -m mypy tinygrad/`;
  lint `python -m ruff check .`
- Perf claims need before/after tok/s from the T0.3 harness on named hardware. Upstream PRs:
  one small lever each, hand-verified — upstream has reverted AI-generated slop before.
- Don't remove the deliberate `.contiguous()` in the MoE expert path (`tinygrad/llm/model.py:27,129`).
