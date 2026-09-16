# Fork rules (arttarawork/tinygrad, 2026-09-13) — these override the upstream notes below for any agent working in this tree
#
# - NEVER `-n12` / `-n <k>` on this 36 GB Mac: one pytest file per process, serially (a kernel panic came from overlapping test waves).
#   `CHECK_OOB=1 DEV=CPU PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python -m pytest <one file> -x -q` (no bare `python`).
# - Never run mypy or pylint over the whole tree unless explicitly asked; ruff on the files you touched is fine.
# - Never touch the model server on :8081, any `TinyGPU … server` process, tmux sessions, launchctl, ~/.hermes, ~/models, or other worktrees.
# - Read CLAUDE.md in this tree first; it is the source of truth. Commit locally on a task branch; never push; never open PRs.

# Notes

- Run tests with `-n12` for speed (e.g. `python -m pytest test/null/test_dtype.py -x -q -n12`)
- Run `python -m mypy tinygrad/` to typecheck
- Run `python -m ruff check .` to lint
- Read `./tinygrad/viz/README.md` for profiling and debugging rewrite rules
- Do not do amend commits. Always do a new commit if a force push to origin would be required.
