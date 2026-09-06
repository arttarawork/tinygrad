import atexit, subprocess
from tinygrad.device import CompileError, CompileTransportError

# T4.31/T4.43: Renderer.__reduce__ (renderer/__init__.py) rebuilds a Compiler from scratch on every
# multiprocessing unpickle -- cheap for every backend that computes in-process (Metal/NAK/LLVM all
# define a plain `__reduce__` for exactly this). The out-of-process compile-server backends are the
# odd ones out: NVRTCCompiler/NVPTXCompiler's __init__ spawns a `docker run` subprocess on OSX
# (T4.31), and QCOMCompiler's __init__ spawns a qemu/docker compile-server whenever
# platform.machine() != "aarch64" (T4.43) -- which is effectively every dev/CI host, since Darwin
# arm64 reports platform.machine() == 'arm64', not 'aarch64'. Without this cache, every BEAM
# candidate handed to a worker (codegen/opt/search.py's pool.imap_unordered, one task per candidate)
# unpickled a fresh Renderer and spawned its own server, used once and abandoned -- hundreds within
# seconds under a real search. Keyed and reaped per-process: each worker process gets its own dict
# and its own atexit hook, so a recycled worker's dead entries never leak into a new one.
_server_cache: dict[tuple, subprocess.Popen] = {}
def _reap_servers() -> None:
  for proc in _server_cache.values():
    if proc.poll() is None: proc.kill()
atexit.register(_reap_servers)

def _get_server(compiler, cmd:str, arch:str, *args) -> subprocess.Popen:
  key = (type(compiler).__name__, arch, args)
  if (proc:=_server_cache.get(key)) is None or proc.poll() is not None:
    proc = _server_cache[key] = compiler.server(cmd, arch, *args)
  return proc

def _evict_server(compiler, arch:str, args:tuple) -> None:
  if (dead:=_server_cache.pop((type(compiler).__name__, arch, args), None)) is not None:
    # T4.45: reap the evicted process instead of leaking it -- guard kill() with poll() (mirrors
    # _reap_servers above) since killing an already-exited pid can raise; wait() unconditionally so
    # a process that's merely pipe-dead (container alive, transport broken) doesn't zombie until the
    # atexit handler eventually runs.
    if dead.poll() is None: dead.kill()
    dead.wait()

def _compile_with_retry(compiler, src:str, cmd:str, arch:str, *args) -> bytes:
  # one retry with a respawned server on a dead transport -- never retries a genuine compile error
  # (CompileError proper, not the CompileTransportError subclass), which propagates immediately.
  try: return compiler.compile_server(src, compiler.compiler_process)
  except CompileTransportError:
    _evict_server(compiler, arch, args)
    compiler.compiler_process = _get_server(compiler, cmd, arch, *args)
    return compiler.compile_server(src, compiler.compiler_process)
  except CompileError: raise  # a fully-formed reply: the pipe is still in sync
  except BaseException:
    # T6.3 (T4.54 root cause): anything else that interrupts the exchange -- BEAM's SIGALRM BeamCompileTimeout mid-read is the
    # one that happens -- leaves THIS request's reply (or the rest of it) in the pipe. Reusing the server would pair every later
    # request with the previous reply, and compile_cached would store that foreign cubin under the new source: the poisoned
    # entries then launch with the wrong grid/indexing and fault the GPU deterministically on every run (2026-09-06, the pooled
    # 27B at 128k: 2 of the 85 launched candidates of one kernel had another candidate's cubin cached). Drop the server instead.
    _evict_server(compiler, arch, args)
    raise
