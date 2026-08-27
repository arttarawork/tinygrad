# T4.40c / RCA §7 — supervised hardware verification runbook

**Artur authorized autonomous `DEV=NV*` execution (2026-08-26 ~20:45): Claude runs this protocol and reports; Artur is reachable during the run. Tiny model / trivial ops
only. Abort the whole session on ANY unexpected fault: stop, touch nothing, `pkill NOTHING`, collect the
unified log (`/usr/bin/log show --last 5m --predicate 'eventMessage CONTAINS[c] "tinygpu"'`), and read the
RCA before the next step.** All work is committed+pushed first (this branch `task/T4.40c-halt-verify`).

## What this session decides
1. **H2 — does an FLR alone halt the previous GSP-RM core?** The one genuine unknown. If yes, 40-3's falcon
   reset is belt-and-suspenders; if no, it is load-bearing. Either way the fix is safe; this tells us *why*.
2. **H1 / H4 / H3** confirm on real silicon what the fake-register tests assert: healthy path untouched (H1),
   the T4.37/40-2 fault-path clear (H4), and the 40-1 failed-init clear (H3).
3. **Outcome:** if H1-H4 pass, T4.40c is cleared to merge (drop the "DO NOT MERGE" and open PR #6).

## Preconditions (check first, no GPU yet)
- `date` ≥ 22:00, and Hermes idle: `curl -s localhost:8080/slots | python3 -c "import json,sys;print('busy',any(s['is_processing'] for s in json.load(sys.stdin)))"` → `busy False`. (Tiny ops run fine beside llama-server; it need not be stopped for H1-H4.)
- Exactly one real server, or zero: `for p in $(pgrep -f 'TinyGPU.*server'); do ps -o command= -p $p | grep -q /Applications/TinyGPU.app && echo "$p"; done` → 0 or 1 line.
- `colima` may stay stopped — every step uses the **NAK lane** (`DEV=NV:NAK`), no docker.
- On the branch: `git -C /Users/artur/Documents/tinygrad-t440c log --oneline -1` → the T4.40c head.
- `WT=/Users/artur/Documents/tinygrad-t440c ; PY=/Users/artur/Documents/tinygrad/.venv/bin/python`

## H1 — baseline (healthy open/close, MASTER untouched)  [~1 min]
```
cd $WT && DEV=NV:NAK PYTHONPATH=. $PY extra/t440c/halt_probe.py h1
```
**Expect:** `MASTER=1` on open, trivial op prints `[2.0, 3.0, 4.0]`, `MASTER=1` after. Server count still 1.
**Abort if:** MASTER reads 0 on a healthy device, or any `Device fault detected`.

## H2 — FLR semantics (the unknown)  [~1 min, reads only]
Needs a *fresh* client that takes the WPR2-up reset path (i.e. a GSP-RM session already resident from H1).
Immediately after H1, with the env flag on:
```
cd $WT && DEV=NV:NAK NV_HALT_DEBUG=1 DEBUG=2 PYTHONPATH=. $PY -c "from tinygrad import Tensor; print((Tensor([1.,2.])+1).tolist())" 2>&1 | grep -E "WPR2|T4.40c H2|result|\[3"
```
**Read off:** `[T4.40c H2] active_stat after FLR, pre-falcon-reset = X`.
- **X=0** → the FLR alone halts the core. Record: "40-3 falcon-reset is redundant on GA102 (FLR suffices); kept as defense-in-depth + for the timeout guard." (still keep the fix — it also handles a wedged core the FLR *doesn't* halt.)
- **X=1** → the FLR does NOT halt it; the falcon reset is what does. **This is the panic-relevant finding** — it means the pre-T4.40c blind `sleep(0.1)`+set-MASTER genuinely raced a live core, and 40-3 closes it. Then `active_stat after falcon reset = 0 (verify passed)` confirms the fix works on silicon.
**Abort if:** the WPR2 branch never prints (no reset happened → not a fresh-after-resident client; re-run after H1), or `active_stat` never reaches 0 (the `wait_cond` will raise `TimeoutError` with MASTER still clear — that is the fix refusing to enable a non-halted core; STOP and record, do not retry blindly).

## H4 — fault path still clears MASTER (T4.37 + 40-2)  [~2 min]  — reuses the t437 harness
```
cd $WT && DEV=NV:NAK PYTHONPATH=. $PY /Users/artur/Documents/tinygrad-t437/extra/t437/fault_repro.py
```
**Expect:** `MASTER before: 1` → `fault raised: ...` → `MASTER after fault: 0`, exit 0.
**Do NOT** run the t437 step-B `pkill` script — the never-kill rule is now permanent (route decision). H4 is the fault→clear readout only; the process exiting cleanly afterward is the 40-2 path.
**Abort if:** exit 2 (MASTER still set after fault).

## H3 — failed-init clears MASTER (40-1, the panic-2 fix)  [~2 min, OPTIONAL]
40-1 already has strong fake-test coverage; this is the on-silicon confirmation, run only if H1/H2/H4 were clean.
```
cd $WT && DEV=NV:NAK PYTHONPATH=. $PY extra/t440c/halt_probe.py h3
```
The probe forces `NV_GSP.init_sw` to raise (after `_early_ip_init` set MASTER), catches it, and prints a
reminder. Then read MASTER back **by hand** (the probe deliberately does not re-open a second session):
```
# in the SAME shell, right after:  a one-shot config read of the NV function's PCI_COMMAND
cd $WT && DEV=NV:NAK PYTHONPATH=. $PY -c "
from tinygrad import Device
d = Device['NV']; import tinygrad.runtime.autogen.pci as pci
print('MASTER now =', (d.iface.pci_dev.read_config(pci.PCI_COMMAND,2)>>2)&1)"
```
**Expect:** the forced-failure open raises; the follow-up open succeeds (fresh client re-enables MASTER) and reads `MASTER now = 1`. The proof that 40-1 fired is in the failed open's own teardown (bus-master cleared before the process would exit); if you want the direct readout, watch for `MASTER` in the failed run's traceback path or add `NV_HALT_DEBUG=1`.
**Abort if:** the follow-up open itself faults or the machine misbehaves.

## Close-out
- All four clean → edit `T4.40c_NOTES.md`: replace "DO NOT MERGE" with the H2 finding + "hardware-verified <date>", remove the env-gated `NV_HALT_DEBUG` prints (or keep them — they're harmless and useful), then open PR #6 (`task/T4.40c-halt-verify` → master), CI, merge. The client-side remediation is then 100% complete and merged.
- Record the H2 value and any surprise in `T4.40c_NOTES.md` §6 and the TASKS T4.40c row.
- Server count 1 (or 0) at exit; `git status` clean; nothing pushed that wasn't intended.
- Then the dock is cleared for measurement: T4.35 runs 2-3 → T4.34 capture → T4.29 nvcc row → the M3 flagship.
