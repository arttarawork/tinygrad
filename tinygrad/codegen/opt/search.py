import math, time, traceback, signal
from collections import Counter
from dataclasses import replace
from tinygrad.uop.ops import sym_infer, AxisType, UOp, Ops
from tinygrad.uop.render import pyrender
from tinygrad.device import Device, Buffer
from tinygrad.helpers import prod, flatten, DEBUG, CACHELEVEL, diskcache_get, diskcache_put, getenv, colored, time_to_str
from tinygrad.helpers import IGNORE_BEAM_CACHE
from tinygrad.codegen.opt import Opt, OptOps, KernelOptError
from tinygrad.engine.realize import time_call
from tinygrad.engine.worker import get_worker_pool, terminate_worker_pool
from tinygrad.codegen import to_program
from tinygrad.codegen.opt.postrange import Scheduler

actions = [Opt(op=OptOps.UPCAST, axis=axis, arg=amt) for amt in [0,2,3,4,5,7] for axis in range(8)]
actions += [Opt(op=OptOps.UNROLL, axis=axis, arg=amt) for amt in [0,4,7] for axis in range(5)]
actions += [Opt(op=OptOps.LOCAL, axis=axis, arg=amt) for amt in [2,3,4,8,13,16,29] for axis in range(6)]
actions += [Opt(op=OptOps.GROUPTOP, axis=axis, arg=amt) for amt in [13,16,28,29,32,49,64,256] for axis in range(3)]
actions += [Opt(op=OptOps.GROUP, axis=axis, arg=amt) for amt in [0,4,8,16] for axis in range(3)]
if getenv("BEAM_PADTO", 0): actions += [Opt(op=OptOps.PADTO, axis=axis, arg=amt) for amt in [32] for axis in range(7)]
actions += [Opt(op=OptOps.LOCAL, axis=0, arg=32), Opt(op=OptOps.LOCAL, axis=6, arg=2)]
actions += [Opt(op=OptOps.TC, axis=0, arg=(-1, 0, getenv("TC", 1)))]
# covers resnet kernels (3 global * 3 reduce)
actions += [Opt(op=OptOps.TC, axis=axis, arg=(-1, getenv("TC_OPT", 2), getenv("TC", 1))) for axis in range(9)]
actions += [Opt(op=OptOps.SWAP, axis=axis_0, arg=axis_1) for axis_0 in range(5) for axis_1 in range(axis_0+1, 5)]
actions += [Opt(op=OptOps.THREAD, axis=axis, arg=amt) for amt in [2,3,4,5,8,12,16,24,32,64] for axis in range(3)]
if getenv("NOLOCALS"): actions += [Opt(op=OptOps.NOLOCALS)]

def get_test_global_size(global_size, max_global_size, var_vals):
  test_global_size = [sym_infer(sz, var_vals) for sz in global_size]
  input_size = prod(test_global_size)
  while prod(test_global_size) > max_global_size:
    for j in range(len(global_size)-1,-1,-1):
      if test_global_size[j] > 16:
        test_global_size[j] //= 2
        break
  return test_global_size, input_size / prod(test_global_size)

def _time_program(prg:UOp, var_vals:dict[str, int], rawbufs:list[Buffer], early_stop:float|None=None,
                  allow_test_size:int=True, max_global_size:int|None=65536, clear_l2=False, cnt=3, name="test", dev_timeout=False) -> list[float]:
  # T4.53: per-candidate LAUNCH log, off by default. `name` was a dead param (no call site ever passed it,
  # always defaulted to "test") -- reused here rather than adding a new kwarg. beam_search's call site packs
  # in the AST id + shape + applied_opts; this print fires right before the candidate's actual GPU launch,
  # so the last LAUNCH lines before a fault-abort name the launches that immediately preceded it (the true
  # faulting candidate may be one of these, not just the one the abort exception names -- see T4.47_RCA.md).
  if BEAM_LAUNCH_LOG: print(f"LAUNCH {time.time():.3f} {name}", flush=True)
  timeout = int(early_stop * 1e3) if dev_timeout and early_stop is not None and early_stop < math.inf else None
  factor = 1
  if allow_test_size and max_global_size is not None:
    global_size, factor = get_test_global_size(prg.arg.global_size, max_global_size, var_vals)
    prg = prg.replace(arg=replace(prg.arg, global_size=tuple(global_size)))
  call = prg.call(*[UOp.from_buffer(b) for b in rawbufs])
  tms, timer = [], time_call(call, var_vals, timeout=timeout, clear_l2=clear_l2)
  for _ in range(cnt):
    try: tms.append(next(timer) * factor)
    except AssertionError: return [math.inf] * cnt
    if early_stop is not None and early_stop < min(tms): break
  return tms

# T4.46: distinguishable failure-cause names for beam_search's WARNING/Counter (previously every cause here
# collapsed onto a generic exception name). Both stay RuntimeError subclasses so the existing
# `except RuntimeError` branch in _try_compile keeps catching them exactly as it already catches plain RuntimeError.
class BeamCompileTimeout(RuntimeError): pass  # SIGALRM fired before to_program finished (see timeout_handler)
class BeamUopLimit(RuntimeError): pass        # candidate's uop count >= BEAM_UOPS_MAX
def timeout_handler(signum, frame):
  if DEBUG >= 2: print("*** BEAM COMPILE TIMEOUT")
  raise BeamCompileTimeout()

def _try_compile(x:tuple[int,Scheduler]) -> tuple[int, tuple[UOp, float]|None, str|None]:
  if hasattr(signal, "alarm"):
    signal.signal(getattr(signal, 'SIGALRM'), timeout_handler)
    # set timeout
    signal.alarm(getenv("BEAM_TIMEOUT_SEC", 10))
  ret, exc_name = None, None
  try:
    st = time.perf_counter()
    ast, dev = x[1].copy().get_optimized_ast(name_override="test"), x[1].ren.target.device
    prg = to_program(ast.substitute({p: p.replace(arg=replace(p.arg, device=dev)) for p in ast.toposort() if p.op is Ops.PARAM}), x[1].ren)
    et = time.perf_counter() - st
    uops = prg.src[1].src
    if len(uops) >= (uops_max:=getenv("BEAM_UOPS_MAX", 3000)) > 0:
      if getenv("BEAM_LOG_SURPASS_MAX"): print(f"too many uops. {len(uops)=}, {uops_max=}")
      raise BeamUopLimit("too many uops")
    ret = (prg, et)
  except RuntimeError as e:
    exc_name = type(e).__name__
    if DEBUG >= 4: traceback.print_exc()
  except Exception as e:
    exc_name = type(e).__name__
    if getenv("BEAM_STRICT_MODE"): raise e
  finally:
    if hasattr(signal, "alarm"): signal.alarm(0)
  return x[0], ret, exc_name

def _ensure_buffer_alloc(bufs:list[Buffer]) -> list[Buffer]: return [buf.ensure_allocated() if buf is not None else buf for buf in bufs]

def _device_faulted(dev) -> bool:
  """T4.48 F1 (T4.47_RCA.md): duck-typed device-fault check. Only NV/AMD's HCQ ifaces model a GPU fault at all,
  always at iface.dev_impl.is_err_state -- every other backend (CPU/METAL/CUDA/NULL/...) has no `iface`, or an
  iface with no `dev_impl.is_err_state`, so the chained getattr below degrades to a plain False there, never an
  AttributeError. Drains first via iface.sleep(0) where the iface exposes it: NV's PCIIface.sleep (ops_nv.py)
  is ALSO T4.37's clear-bus-master-on-fault site, so calling it (instead of re-deriving a second drain here)
  reuses that safety-critical logic rather than duplicating it. sleep(0), not sleep(200): NV's `timeout` arg is
  entirely unused (it just drains+checks unconditionally) but AMD's iface honors it as a real IRQ-poll timeout,
  so 0 keeps this a probe -- never a wait -- on every backend that might reach it. iface.sleep() itself already
  raises RuntimeError on a real fault, so that's treated as "faulted" too rather than letting its generic
  message escape here; the caller composes the richer, candidate-naming message.
  """
  iface = getattr(dev, "iface", None)
  if iface is not None and hasattr(iface, "sleep"):
    try: iface.sleep(0)
    except RuntimeError: return True
  return bool(getattr(getattr(iface, "dev_impl", None), "is_err_state", False))

# *** external API ***

# get dictionary of all possible actions
def get_kernel_actions(s:Scheduler, include_0=True, max_up:int|None=None) -> dict[int, Scheduler]:
  acted, max_up, max_lcl = {0:s} if include_0 else {}, getenv("BEAM_UPCAST_MAX", 256) if max_up is None else max_up, getenv("BEAM_LOCAL_MAX", 1024)
  kernel_actions = actions.copy()

  for i,a in enumerate(kernel_actions):
    if a.axis is not None and a.op is not OptOps.TC:
      try: ax = s.real_axis(a.op, a.axis)
      except KernelOptError: continue
      if (ax >= s.shape_len) or (s.full_shape[ax] == a.arg and Opt(a.op, a.axis, 0) in kernel_actions): continue
    s2 = s.copy()
    try:
      s2.apply_opt(a)
      up, lcl, tc_up = 1, 1, prod(tc.dims)//tc.threads if hasattr(s2, 'tensor_core') and (tc:=s2.tensor_core) else 1
      for x,t in zip(s2.full_shape, s2.axis_types):
        if t in (AxisType.UPCAST, AxisType.UNROLL): up *= x
        elif t in (AxisType.WARP, AxisType.LOCAL, AxisType.GROUP_REDUCE): lcl *= x
      if up//tc_up > max_up or lcl > max_lcl:
        if getenv("BEAM_LOG_SURPASS_MAX"): print(f"too many upcast/local. {up//tc_up=}, {max_up=}, {lcl=}, {max_lcl=}")
        continue
      acted[i+1] = s2
    except KernelOptError: pass
  return acted

BEAM_DEBUG = getenv("BEAM_DEBUG")
BEAM_LAUNCH_LOG = getenv("BEAM_LAUNCH_LOG", 0)  # T4.53: gate for _time_program's per-candidate LAUNCH log
def beam_search(s:Scheduler, rawbufs:list[Buffer], var_vals:dict[str,int], amt:int, allow_test_size=True, disable_cache=IGNORE_BEAM_CACHE.value):
  key = {"ast": s.ast.key, "amt": amt, "allow_test_size": allow_test_size, "device": s.ren.target.device, "suffix": s.ren.suffix}
  if not disable_cache and CACHELEVEL >= 1 and (val:=diskcache_get("beam_search", key)) is not None:
    ret = s.copy()
    for o in val[len(s.applied_opts):]: ret.apply_opt(o)
    return ret

  beam: list[tuple[Scheduler, float]] = [(s, float("inf"))]
  seen_libs = set()

  pool = get_worker_pool()

  min_progress = getenv("BEAM_MIN_PROGRESS", 0.01)/1e6
  if BEAM_DEBUG:
    print("BEAM_SEARCH:")
    print(pyrender(s.ast.replace(arg=None)))
  if DEBUG >= 2: print(f"   0.00s:                from   1 ->   1 actions {s.colored_shape()}")

  try:
    rawbufs = _ensure_buffer_alloc(rawbufs)
    exiting, st = False, time.perf_counter()
    dev = Device[s.ren.target.device]
    while not exiting:
      candidates: list[Scheduler] = flatten([get_kernel_actions(si, include_0=False).values() for si,_ in beam])
      timed: list[tuple[Scheduler, float]] = []
      fails: Counter[str] = Counter()
      least_compute_ops = math.inf
      for i, proc, exc_name in ((map if pool is None else pool.imap_unordered)(_try_compile, enumerate(candidates))):
        if proc is None:
          fails[exc_name or "unknown"] += 1
          continue
        prg, compile_et = proc
        if (lib:=prg.src[3].arg) in seen_libs: continue
        # filter out kernels that use 1000x more compute than the smallest
        estimates = prg.src[0].arg.estimates
        least_compute_ops = min(this_compute_ops:=sym_infer(estimates.ops if estimates is not None else 0, var_vals), least_compute_ops)
        if least_compute_ops*1000 < this_compute_ops:
          if getenv("BEAM_LOG_SURPASS_MAX"): print(f"too much compute. {this_compute_ops} when least is {least_compute_ops}")
          continue
        seen_libs.add(lib)
        try: tms = _time_program(prg, var_vals, rawbufs, early_stop=beam[0][1]*3 if len(beam) else 1.0,
                                 allow_test_size=allow_test_size, clear_l2=hasattr(dev, 'invalidate_caches'),
                                 dev_timeout=getenv("BEAM_DEV_TIMEOUT", 1),
                                 # T4.53: AST id (structural hash, stable across all candidates/rounds of THIS
                                 # kernel) + the kernel's pre-search shape (recognizable at a glance) + this
                                 # candidate's applied_opts. Conditional (not computed when the log is off) so
                                 # this costs nothing in the default/hot path.
                                 name=f"{s.ast.key.hex()[:12]} {s.colored_shape()} {candidates[i].applied_opts}" if BEAM_LAUNCH_LOG else "test")
        except Exception as e:
          if BEAM_DEBUG: print(f"BEAM failed for opts: {candidates[i].applied_opts}\n{e}")
          if isinstance(e, RuntimeError):
            # coordinator fold-in (T4.48, same code region as F1): hcq.py's HCQSignal.wait raises a bare
            # RuntimeError("Wait timeout: ...") that was otherwise indistinguishable from every other
            # RuntimeError in this Counter. String-matched, not a bespoke exception class -- hcq.py:296 is
            # generic runtime code shared far outside BEAM, out of scope to touch here (T4.46 instead added
            # real subclasses at its own two call sites inside this file, _try_compile's; this one can't
            # follow that pattern without editing hcq.py). Fragile if that message wording ever changes;
            # low blast radius -- only renames a Counter/WARNING entry, nothing branches on the name itself.
            fails["BeamDeviceTimeout" if str(e).startswith("Wait timeout") else type(e).__name__] += 1
            # T4.48 F1 (T4.47_RCA.md): beam_search abandons a slow candidate here by design -- nothing
            # cancels its still-running kernel, so a candidate that FAULTS the GPU (its completion signal
            # never fires, same as a merely-slow one) raises this identical exception and was previously
            # swallowed identically. Check now, right after the failure, whether the device is actually
            # faulted (not just slow) and abort the whole search immediately if so, instead of grinding out
            # the remaining rounds against a dead device producing all-inf results.
            if _device_faulted(dev):
              raise RuntimeError(f"BEAM: device fault detected during search -- aborting rather than "
                                  f"continuing against a faulted device. Failing candidate's applied_opts: "
                                  f"{candidates[i].applied_opts}. Original error: {e}") from e
            continue
          raise
        timed.append((candidates[i], min(tms)))
        if BEAM_DEBUG > 1:
          print(f"{time.perf_counter() - st:7.2f}s: {i:5d} {len(prg.src[1].src):5d} uops",
                f"{time_to_str(compile_et, w=12)} compile/{time_to_str(timed[-1][1], w=12)} run",
                f"      {len(timed):4d}/{len(candidates):4d}         {timed[-1][0].colored_shape()}")
        elif DEBUG >= 2:
          print(f"\r{time.perf_counter() - st:7.2f}s: {time_to_str(timed[-1][1], w=12)}",
                f"      {len(timed):4d}/{len(candidates):4d}         {timed[-1][0].colored_shape()}\033[K", end="")

      # done
      opts = sorted(timed, key=lambda x: x[1])
      if candidates and not opts:
        print(colored(f"WARNING: BEAM found no viable candidate this round ({len(candidates)} tried, {sum(fails.values())} failed: "
                       f"{dict(fails)}) for {beam[0][0].colored_shape()}", "red"))
      exiting = len(opts) == 0 or (opts[0][1] < min_progress) or (len(beam) > 0 and ((beam[0][1]-opts[0][1]) < min_progress))
      if not exiting: beam = opts[:amt]
      elif len(opts) > 0 and opts[0][1] < beam[0][1]: beam = opts[:1]
      if DEBUG >= 2:
        print(f"\r{time.perf_counter() - st:7.2f}s:", colored(time_to_str(beam[0][1], w=12), "green" if exiting else None),
              f"from {len(candidates):3d} -> {len(opts):3d} actions\033[K", beam[0][0].colored_shape())
  except KeyboardInterrupt as e:
    terminate_worker_pool()
    raise e

  if beam[0][0] is s and candidates:
    # every round found zero viable candidates (see the per-round WARNING above): don't hand back an untuned kernel silently,
    # apply the same heuristics used when BEAM isn't requested at all (postrange.apply_opts' beam==0 path)
    print(colored(f"WARNING: BEAM found no working action for {s.colored_shape()}, falling back to hand_coded_optimizations", "red"))
    from tinygrad.codegen.opt.heuristic import hand_coded_optimizations
    beam[0] = (hand_coded_optimizations(beam[0][0].copy()), beam[0][1])
  # T4.39: an inf best time means the search never produced an empirically-validated winner -- either the
  # total-failure fallback above (whose score is still the untouched seed's inf) or a candidate whose only
  # timing attempt hit _time_program's AssertionError path (search.py:50). Don't persist that: a later run
  # would otherwise silently replay an unvalidated kernel forever with none of the WARNINGs above.
  if CACHELEVEL >= 1 and not math.isinf(beam[0][1]): diskcache_put("beam_search", key, beam[0][0].applied_opts)
  if BEAM_DEBUG: print(f"BEAM_SEARCH: final tm={time_to_str(beam[0][1], w=0)}, applied_opts={beam[0][0].applied_opts}")
  # T4.48 F3 (T4.47_RCA.md): bound the lifetime of any candidate this search abandoned-but-never-cancelled (see
  # the RuntimeError handler above) -- wait for the device's timeline to fully drain before handing the winning
  # kernel back, so a caller that immediately frees/evicts rawbufs can never race a still-running abandoned
  # kernel. A fault surfacing only now (never during the search loop's own per-candidate check) would otherwise
  # raise HCQCompiled.synchronize's bare message; name it as search cleanup instead.
  try: dev.synchronize()
  except RuntimeError as e: raise RuntimeError(f"BEAM: device faulted during search cleanup (synchronize after "
                                                f"search) for {s.colored_shape()}: {e}") from e
  return beam[0][0]
