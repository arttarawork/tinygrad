"""
T4.49 Phase 2 -- ad-hoc runtime instrumentation (NOT committed to tinygrad/, see T4.49_NOTES.md).

Auto-imported by Python's `site` module at interpreter startup (this file's directory must be on
PYTHONPATH) -- works for `python -m tinygrad.llm ...` where PYTHONSTARTUP would NOT fire (that only
runs for interactive sessions).

Patches tinygrad.codegen.opt.search.{_time_program,beam_search} in place to record, for every BEAM
run in this process:
  - one CSV row per candidate timed (T449_CSV): search id, sequence within that search, wall
    timestamp, kernel global_size/uop-count fingerprint, the early_stop*3 threshold BEAM used,
    whether this run synced before timing, the outcome (ok / RuntimeError "abandoned"), and the
    measured tms.
  - one CSV row per beam_search() call (T449_PICKS_CSV): the ast key, how many candidates got
    timed, the final winning applied_opts, and search wall time.

Run A (stock): T449_SYNC_BEFORE unset -- pure observation, no behavior change.
Run B: T449_SYNC_BEFORE=1 -- before every candidate's _time_program call, synchronize() the
candidate's device first (drains anything left running from a prior abandonment). Everything else
(the CSV writes) is identical in both runs so the recording overhead itself doesn't skew the A/B.

Required env: T449_CSV, T449_PICKS_CSV, T449_RUN (label, e.g. "A" or "B").
Optional env: T449_SYNC_BEFORE (any non-empty value enables the pre-candidate sync).
"""
import csv, os, time, threading

try:
  _CSV, _PICKS_CSV, _RUN = os.environ["T449_CSV"], os.environ["T449_PICKS_CSV"], os.environ.get("T449_RUN", "?")
except KeyError:
  pass  # not a T4.49 run (e.g. a plain interpreter) -- do nothing
else:
  from tinygrad.codegen.opt import search as _search
  from tinygrad.device import Device

  _SYNC_BEFORE = bool(os.environ.get("T449_SYNC_BEFORE"))
  _orig_time_program, _orig_beam_search = _search._time_program, _search.beam_search
  _lock = threading.Lock()
  _search_id, _seq_in_search = [0], [0]

  _cf = open(_CSV, "a", newline="")
  _cw = csv.writer(_cf)
  if _cf.tell() == 0:
    _cw.writerow(["run", "search_id", "seq", "ts", "global_size", "num_uops", "early_stop_ms",
                  "sync_before", "outcome", "min_tm_ms", "all_tms_ms"])
    _cf.flush()

  _pf = open(_PICKS_CSV, "a", newline="")
  _pw = csv.writer(_pf)
  if _pf.tell() == 0:
    _pw.writerow(["run", "search_id", "ast_key", "num_candidates_timed", "applied_opts", "final_tm_ms", "wall_s"])
    _pf.flush()

  def _wrapped_time_program(prg, var_vals, rawbufs, early_stop=None, allow_test_size=True,
                             max_global_size=65536, clear_l2=False, cnt=3, name="test", dev_timeout=False):
    ts = time.time()
    if _SYNC_BEFORE:
      try: Device[rawbufs[0].device].synchronize()
      except Exception: pass  # never let instrumentation itself change search outcomes
    outcome, tms = "ok", []
    try:
      tms = _orig_time_program(prg, var_vals, rawbufs, early_stop=early_stop, allow_test_size=allow_test_size,
                                max_global_size=max_global_size, clear_l2=clear_l2, cnt=cnt, name=name, dev_timeout=dev_timeout)
      return tms
    except RuntimeError as e:
      outcome = ("RuntimeError:" + str(e)).replace("\n", " ").replace(",", ";")[:200]
      raise
    finally:
      with _lock:
        _seq_in_search[0] += 1
        try: gsize = prg.arg.global_size
        except Exception: gsize = "?"
        try: nuops = len(prg.src[1].src)
        except Exception: nuops = -1
        _cw.writerow([_RUN, _search_id[0], _seq_in_search[0], f"{ts:.6f}", gsize, nuops,
                      f"{(early_stop*1e3) if early_stop is not None else ''}", int(_SYNC_BEFORE),
                      outcome, (f"{min(tms):.6f}" if tms else ""), ";".join(f"{t:.6f}" for t in tms)])
        _cf.flush()

  def _wrapped_beam_search(s, rawbufs, var_vals, amt, allow_test_size=True, disable_cache=_search.IGNORE_BEAM_CACHE.value):
    with _lock:
      _search_id[0] += 1
      _seq_in_search[0] = 0
      sid = _search_id[0]
    t0 = time.time()
    ret = _orig_beam_search(s, rawbufs, var_vals, amt, allow_test_size=allow_test_size, disable_cache=disable_cache)
    wall = time.time() - t0
    with _lock:
      n = _seq_in_search[0]
      _pw.writerow([_RUN, sid, str(s.ast.key), n, str(ret.applied_opts).replace(",", ";"), "", f"{wall:.3f}"])
      _pf.flush()
    return ret

  _search._time_program, _search.beam_search = _wrapped_time_program, _wrapped_beam_search
  print(f"[t449] patched search._time_program + beam_search -> run={_RUN} sync_before={_SYNC_BEFORE} "
        f"csv={_CSV} picks={_PICKS_CSV}", flush=True)
