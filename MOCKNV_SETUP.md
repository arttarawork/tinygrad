# Mock-NV on macOS (T0.4)

Runs the `NV` backend against `test/mockgpu` + gpuocelot PTX emulation on Apple Silicon, so
NV-touching code (`tinygrad/runtime/ops_nv.py`, `MOCKIface`) can be exercised without the eGPU
dock. Verified on this machine (M3 Pro, macOS, arm64) against upstream `af2a43c85`.

**Status: green.** No repo-side code changes were required — everything below worked out of the
box against `af2a43c85`. `extra/setup_mock_nv_osx.sh` was read but **not run** (see "Why not the
in-tree script" below); this doc supersedes it as the working recipe.

## TL;DR

```bash
# one-time: fetch the prebuilt gpuocelot dylib CI actually uses (~2s, 3.2MB, no build)
curl -fL -o /Users/artur/Documents/tinygrad/.venv/lib/libgpuocelot.dylib \
  https://github.com/tinygrad/gpuocelot/releases/download/v0.1.0/libgpuocelot.dylib

# every run: point tinygrad's DLL loader at it via OCELOT_PATH, and target the mock NV+PTX combo
OCELOT_PATH=/Users/artur/Documents/tinygrad/.venv/lib/libgpuocelot.dylib \
DEV=MOCK+NV:PTX PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python -m pytest test/test_tiny.py -x -q
```

Result: `19 passed, 2 skipped in ~3s` (skips are the two pre-existing CL-only image tests, unrelated
to mock).

## Important correction to the stated objective command

The literal objective command, **`DEV=NV ... pytest test/test_tiny.py`, does NOT work** and is not
what CI runs. Verified:

```
$ DEV=NV PYTHONPATH=. .venv/bin/python -m pytest test/test_tiny.py -x -q
...
FileNotFoundError: [Errno 2] No such file or directory: '/dev/nvidiactl'
...
RuntimeError: NV:0 does not exist (0 devices available)
FAILED test/test_tiny.py::TestTiny::test_beam - ExceptionGroup: No interface ...
```

Root cause (`tinygrad/device.py:376` `_select_iface`, and `Target.parse` in `tinygrad/helpers.py`):
`NVDevice.ifaces = [NVKIface, PCIIface, MOCKIface]` (`tinygrad/runtime/ops_nv.py:586`). When the
`DEV=` interface field is empty (bare `DEV=NV`), `_select_iface` explicitly **excludes any iface
whose name starts with "MOCK"** ("never fallback to mock ifaces") and tries `NVKIface` (real
`/dev/nvidiactl`) then `PCIIface` (real PCI enumeration) — both fail on a Mac with no NVIDIA
hardware, and `MOCKIface` is never reached.

**You must opt into the mock interface explicitly**: `DEV=MOCK+NV` or `DEV=MOCK+NV:PTX`. This
matches `.github/workflows/platform.yml`'s `unittestmacosmock` job, which is CI's ground truth for
this exact scenario (macOS + mock NV) — it uses `DEV: "MOCK+NV:PTX"`, not `DEV=NV`.

Both `DEV=MOCK+NV` and `DEV=MOCK+NV:PTX` were verified to pass identically here (19 passed, 2
skipped either way) — `:PTX` isn't strictly required locally because the default renderer probe
order is `[CUDARenderer, PTXRenderer, NVCCRenderer, NAKRenderer]`
(`tinygrad/runtime/ops_nv.py:626`), and `CUDARenderer.__init__` eagerly constructs an
`NVRTCCompiler` (`tinygrad/renderer/cstyle.py:400`) which throws immediately on macOS (no libnvrtc),
so `select_first_inited` silently falls through to `PTXRenderer` anyway. **Still, pin `:PTX`
explicitly** — it's what CI does, it's deterministic, and it doesn't depend on CUDARenderer
continuing to fail-fast in future refactors.

## Why not the in-tree script (`extra/setup_mock_nv_osx.sh`)

Read before running, per task instructions. It:
- `brew install`s `cmake ninja llvm@15 zlib glew flex bison boost zstd ncurses` (large, includes a
  non-default-linked `llvm@15`),
- clones `gpuocelot/gpuocelot` and builds `libgpuocelot.dylib` from source via cmake/ninja,
- **`sudo cp libgpuocelot.dylib /usr/local/lib/`** — writes outside homebrew/pip/venv/repo, needs
  sudo.

That last step is out of the constrained scope for this task (system dir + sudo), so per
instructions this was **not run** — flagging it here instead.

More importantly, **it's not what CI does and isn't necessary**: `.github/actions/setup-tinygrad/action.yml`
(the `ocelot: 'true'` input, lines ~252-258) just downloads a **prebuilt** binary:

```bash
sudo curl --output-dir /usr/local/lib -fLO \
  https://github.com/tinygrad/gpuocelot/releases/download/v0.1.0/libgpuocelot.${{ runner.os == 'Linux' && 'so' || 'dylib' }}
```

i.e. `https://github.com/tinygrad/gpuocelot/releases/download/v0.1.0/libgpuocelot.dylib` for macOS —
the tinygrad org publishes a ready-to-use arm64 Mach-O dylib. No toolchain, no build, no sudo
needed: `tinygrad/runtime/support/c.py`'s `DLL.findlib()` (used by `test/mockgpu/helpers.py`'s
`ptx_run` binding, `gpuocelot_lib = c.DLL("ocelot", "gpuocelot")`) honors a per-library env override
**`OCELOT_PATH=<exact file path>`**, so the dylib can live anywhere (here: `.venv/lib/`, which is
gitignored and squarely "pip-venv" scope) with zero system-directory writes and zero sudo. CI's own
`sudo cp ... /usr/local/lib` is just so the loader's default darwin search path
(`/opt/homebrew/lib`, framework dirs, plus `LD_LIBRARY_PATH`/`/usr/local/lib` from the shared posix
list) finds it without an env var — not a functional requirement.

`extra/setup_mock_nv_osx.sh` looks like a stale/manual alternative to the CI recipe (predates the
GitHub Releases artifact, or intended for contributors who want to rebuild gpuocelot itself). Not
touched, not needed for this task.

## Exact steps (reproducible in one shot)

```bash
# 1. branch from the baseline this doc was verified against
git checkout -b task/T0.4-mocknv af2a43c85

# 2. fetch the prebuilt dylib CI publishes (arm64 Mach-O, 3.2MB, ~2s on a normal link)
curl -fL -o /Users/artur/Documents/tinygrad/.venv/lib/libgpuocelot.dylib \
  https://github.com/tinygrad/gpuocelot/releases/download/v0.1.0/libgpuocelot.dylib
# sha256: 5106c998c795a36dec79eb7b2aae324a93d1338236d36eeaae232649ec457663 (integrity check only;
# re-verify against the release page if this ever needs to be re-pinned)

# 3. run the objective test
OCELOT_PATH=/Users/artur/Documents/tinygrad/.venv/lib/libgpuocelot.dylib \
DEV=MOCK+NV:PTX PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python -m pytest test/test_tiny.py -x -q
```

No `brew install`, no C build, no sudo, no system directory writes. Total wall time for setup
(after the repo/venv already exist) is dominated by the ~2s download; the test run itself is ~3s.

## Verbatim result

```
$ OCELOT_PATH=/Users/artur/Documents/tinygrad/.venv/lib/libgpuocelot.dylib DEV=MOCK+NV:PTX \
  PYTHONPATH=. /Users/artur/Documents/tinygrad/.venv/bin/python -m pytest test/test_tiny.py -x -q
.s........s..........                                                    [100%]
19 passed, 2 skipped in 2.96s
```

(The 2 skips are `test_image`/`test_beam_image`: `"image only supported on CL"` — pre-existing,
unrelated to mock.)

## Other CI-equivalent suites verified (functional smoke, not required by the objective)

`unittestmacosmock`'s "ptx job" in `.github/workflows/platform.yml` also runs these under the same
`DEV=MOCK+NV:PTX`; both pass here too:

```bash
OCELOT_PATH=.venv/lib/libgpuocelot.dylib DEV=MOCK+NV:PTX FORWARD_ONLY=1 PYTHONPATH=. \
  .venv/bin/python -m pytest test/device/test_hcq.py -q
# -> 29 passed, 4 skipped, 14 subtests passed in ~2.5s

OCELOT_PATH=.venv/lib/libgpuocelot.dylib DEV=MOCK+NV:PTX FORWARD_ONLY=1 CAPTURE_PROCESS_REPLAY=0 \
  PYTHONPATH=. .venv/bin/python -m pytest test/testextra/test_hevc.py::TestHevc::test_hevc_decode_compile -q
# -> 1 passed in ~1.4s
```

CI sets `CAPTURE_PROCESS_REPLAY: 0` on that job with a `# TODO: failing due to library loading
error` comment — a known CI-side process-replay-capture quirk unrelated to functional correctness;
not encountered here since process replay wasn't exercised.

## Which suites are meaningful under mock

**Functional only — mock-NV timing is meaningless for perf claims** (gpuocelot interprets PTX in
software; the T0.3 perf harness requires named real hardware per the repo's `CLAUDE.md`).

- Meaningful and verified: `test/test_tiny.py`, `test/device/test_hcq.py`,
  `test/testextra/test_hevc.py::TestHevc::test_hevc_decode_compile` — exactly CI's mock-NV
  coverage, all green.
- Likely meaningful but **not exercised in this pass** (out of scope for the stated objective):
  broader `test/backend/*` suites. Grep shows partial/uneven mock-NV support baked in as explicit
  skips, e.g. `test/backend/test_linearizer.py:291` skips a PTX-indexing case under MOCKGPU+PTX
  ("might be ok?"), `test/backend/test_edgecases.py:206` skips a case that hangs gpuocelot, and
  `test/backend/test_ops.py:3347` skips a reduce case as "very slow on MOCKGPU". Treat these as
  known gpuocelot/PTX-emulation edges, not new findings from this task.
- Not meaningful: anything timing-sensitive, anything requiring `PCIIface`/real driver ioctls
  (mock intercepts allocation but not real hardware paths), multi-GPU counts beyond `MOCKIface.count
  = 1`.

## Quirks recap

- `MOCKIface` is `NVDevice.ifaces[2]` and is **opt-in only**: the DEV target's interface field must
  literally start with `"MOCK"` (`tinygrad/device.py` `_select_iface`), so `DEV=MOCK+NV[:PTX]`, not
  `DEV=NV`.
- `OCELOT_PATH=<file>` (or `LD_LIBRARY_PATH=<dir containing (lib)gpuocelot.dylib>`) is honored by
  `tinygrad/runtime/support/c.py`'s generic `DLL.findlib()` — same mechanism as `NVRTC_PATH`,
  `LLVM_PATH`, etc. elsewhere in the codebase. Used here to avoid any sudo/system-dir write.
- CUDARenderer is probed before PTXRenderer when no renderer is pinned, but fails fast on macOS (no
  nvrtc) and falls through automatically — don't rely on this, pin `:PTX`.
- No repo-side patches were needed to make any of this pass on `af2a43c85`.

## Branch

`task/T0.4-mocknv`, based on `af2a43c85`. This file is the only change on the branch.
