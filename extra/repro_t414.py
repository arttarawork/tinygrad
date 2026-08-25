#!/usr/bin/env python3
"""T4.14 repro: OSX docker compile-server short-read truncation.

Compiler.server() (tinygrad/device.py) spawns the OSX docker compile server with
bufsize=0, so proc.stdout is a raw unbuffered pipe. Compiler.compile_server() then
reads the length-prefixed reply with single .read(4) / .read(size) calls. A raw
pipe read is ONE syscall and can return fewer bytes than requested once the reply
crosses a pipe-buffer-sized chunk, so compile() silently returns a truncated lib.

This calls the real, unmodified Compiler.compile() for increasing output sizes and,
right after it returns, drains anything still sitting unread in the pipe. Leftover
bytes prove the reply was longer than what compile() handed back.

Run: PYTHONPATH=. <venv>/bin/python extra/repro_t414.py
"""
import select
from tinygrad.helpers import unwrap, OSX
from tinygrad.runtime.support.compiler_cuda import NVRTCCompiler

assert OSX, "this repro only exercises the OSX docker compile-server path"


def make_src(n: int) -> str:
  body = "\n".join(f"  out[{i}] = a[{i}] * {i}.0f - b[{i}] / {i + 1}.0f + {i}.0f;" for i in range(n))
  return f'extern "C" __global__ void unrolled(float* out, float* a, float* b) {{\n{body}\n}}\n'


def drain(f, timeout=0.5) -> bytes:
  leftover, fd = b"", f.fileno()
  while select.select([fd], [], [], timeout)[0]:
    if not (chunk := f.read(1 << 16)): break
    leftover += chunk
  return leftover


def probe(compiler: NVRTCCompiler, n: int) -> tuple[int, int, Exception | None]:
  try:
    lib = compiler.compile(make_src(n))
  except Exception as e:
    return 0, len(drain(unwrap(compiler.compiler_process.stdout))), e
  return len(lib), len(drain(unwrap(compiler.compiler_process.stdout))), None


def main():
  compiler = NVRTCCompiler("sm_86", ptx=False)
  try:
    for n in (1, 300, 800, 1500, 3000, 6000, 10000):
      got, leftover, exc = probe(compiler, n)
      true_size = got + leftover
      status = f"EXC: {exc!r}" if exc else ("TRUNCATED" if leftover else "ok")
      print(f"n_statements={n:6d}  compile()_returned={got:7d}B  drained_after={leftover:7d}B  true_size={true_size:7d}B  {status}")
  finally:
    compiler.compiler_process.terminate()


if __name__ == "__main__":
  main()
