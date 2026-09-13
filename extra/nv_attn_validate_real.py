"""T4.107a/b: validate + time the NV split-KV decode attention AND the tiled prefill attention (kernels/nv_attn.py) on a real GGUF.

Two passes of the same script, one per attention path, then a compare -- the decode graph is captured once per process (the
@function/JIT trace does not re-key on NV_CUSTOM_ATTN), so the paths cannot be flipped inside one process:
  DEV=NV KV_INT4=1 NV_CUSTOM_QUANT=1 GDN_NV_FUSED_DECODE=1 KQUANT_STAGE=0 NV_CUSTOM_ATTN=0 PYTHONPATH=. python extra/nv_attn_validate_real.py run \
    --model /Users/artur/models/qwen3.8-27b-q4/Qwen3.8-27B-Q4_0.gguf --max-context 65536 --chunk-size 128 --fill 2048,20480,61440 --out generic.npz
  ... NV_CUSTOM_ATTN=1 ... --out custom.npz
  PYTHONPATH=. python extra/nv_attn_validate_real.py compare generic.npz custom.npz
Per fill size N: the cache is filled with N seeded random tokens (chunked prefill through Transformer.__call__ with bound Variables,
exactly generate()'s decode graphs), then `--steps` greedy decode steps are timed (ms/step, first step excluded: it captures the JIT)
and the first step's argmax + top-2 logit gap are recorded. `compare` prints per N: ms/step for both, the speedup, whether the argmax
agrees, and the top-2 gap difference (the kernel's fp32 online softmax vs the generic path's fp16 dequant + SDPA: expect small gaps to
differ in the 1e-2 range, argmax to agree).
T4.107b: after the decode steps at each fill, `--prefill-steps` chunks of `--probe-chunk` tokens (the SERVED chunk width, 32: its own
jit family, so the prefill kernel sees T_pad = 32 exactly as generate() does; the first step captures it and is not timed) are timed
as ms per chunk and the chunk's last-position argmax/top-2 gap recorded under `p<fill>`; `compare` prints both tables. Fill with
--chunk-size 128 at --max-context 65536 (the generic path's (chunk x max_context) score arena OOMs the card at 512 x 131072)."""
import argparse, time
import numpy as np
from tinygrad import Tensor, UOp
from tinygrad.llm.model import Transformer
from extra.flip_rate import reset_recurrent_state, _spec_step, _top1_and_gap

def run(args:argparse.Namespace) -> None:
  model, _ = Transformer.from_gguf(args.model, args.max_context, device_map=args.device_map)
  dev = model.blk[0].device
  fills = [int(n) for n in args.fill.split(",")]
  assert max(fills) + args.steps + args.prefill_steps * args.probe_chunk < args.max_context, "fill + steps must fit max_context"
  v_start_pos, v_toks = UOp.variable("start_pos", 0, model.max_context - 1), UOp.variable("toks", 1, args.chunk_size)
  v_probe = UOp.variable("toks", 1, args.probe_chunk)  # same name (__call__ keys the family on its vmax), the served width
  rng = np.random.default_rng(0)
  ids = rng.integers(100, 20000, size=max(fills), dtype=np.int32)  # any tokens: the numbers, not the text, are compared
  t = Tensor(np.concatenate([ids, np.zeros(model.max_context - len(ids), dtype=np.int32)]).reshape(1, -1), dtype="int32", device=dev)
  results = {}
  for n in fills:
    reset_recurrent_state(model)
    pos, t0 = 0, time.perf_counter()
    while pos < n:
      k = min(args.chunk_size, n - pos)
      sp, nt = v_start_pos.bind(pos), v_toks.bind(k)
      logits = _spec_step(model, t[:, sp:sp + nt], sp, k)
      pos += k
    logits.numpy()
    prefill_s = time.perf_counter() - t0
    top1, gap = _top1_and_gap(logits)
    step_ms = []
    tok = top1
    for i in range(args.steps):
      t1 = time.perf_counter()
      logits = _spec_step(model, Tensor([[tok]], dtype="int32", device=dev), v_start_pos.bind(pos), 1)
      tok, _ = _top1_and_gap(logits)  # .numpy() inside: the sync that makes the timing real
      step_ms.append((time.perf_counter() - t1) * 1000)
      pos += 1
    timed = step_ms[1:] if len(step_ms) > 1 else step_ms
    print(f"fill {n:7d}: prefill {prefill_s:7.1f} s ({n / prefill_s:6.1f} tok/s)  decode {np.mean(timed):7.1f} ms/step"
          f" (min {min(timed):7.1f})  first-step argmax {top1} gap {gap:.4f}", flush=True)
    results[f"{n}"] = np.array([np.mean(timed), min(timed), top1, gap], dtype=np.float64)
    chunk_ms = []
    for i in range(args.prefill_steps):
      t1 = time.perf_counter()
      sp, nt = v_start_pos.bind(pos), v_probe.bind(args.probe_chunk)
      logits = _spec_step(model, t[:, sp:sp + nt], sp, args.probe_chunk)
      ptop1, pgap = _top1_and_gap(logits)
      chunk_ms.append((time.perf_counter() - t1) * 1000)
      pos += args.probe_chunk
    ptimed = chunk_ms[1:] if len(chunk_ms) > 1 else chunk_ms
    print(f"fill {n:7d}: prefill chunk of {args.probe_chunk} at {pos - args.probe_chunk} filled: {np.mean(ptimed):7.1f} ms/chunk"
          f" ({args.probe_chunk / np.mean(ptimed) * 1000:6.1f} tok/s)  last-position argmax {ptop1} gap {pgap:.4f}", flush=True)
    results[f"p{n}"] = np.array([np.mean(ptimed), min(ptimed), ptop1, pgap], dtype=np.float64)
  np.savez(args.out, **results)
  print(f"wrote {args.out}")

def compare(args:argparse.Namespace) -> None:
  a, b = np.load(args.a), np.load(args.b)
  for label, keys in (("decode ms/step", [k for k in a.files if not k.startswith("p")]),
                      ("prefill ms/chunk", [k for k in a.files if k.startswith("p")])):
    if not keys: continue
    print(f"{label:>16} {'A':>10} {'B':>10} {'speedup':>8}  argmax  gap A / gap B")
    for key in sorted(keys, key=lambda k: int(k.lstrip("p"))):
      ma, mb = a[key], b[key]
      agree = 'same' if int(ma[2]) == int(mb[2]) else 'DIFF'
      print(f"{int(key.lstrip('p')):16d} {ma[0]:10.1f} {mb[0]:10.1f} {ma[0] / mb[0]:8.2f}x  {agree:5s}  {ma[3]:.4f} / {mb[3]:.4f}")

def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = parser.add_subparsers(dest="cmd", required=True)
  r = sub.add_parser("run")
  r.add_argument("--model", required=True)
  r.add_argument("--max-context", type=int, default=131072)
  r.add_argument("--fill", default="2048,20480,61440", help="comma-separated filled-context sizes")
  r.add_argument("--steps", type=int, default=6, help="decode steps per fill (the first is the JIT capture and is not timed)")
  r.add_argument("--chunk-size", type=int, default=128, help="fill chunk width (its own jit family)")
  r.add_argument("--probe-chunk", type=int, default=32, help="T4.107b: the served prefill chunk width to time at each fill")
  r.add_argument("--prefill-steps", type=int, default=3, help="T4.107b: prefill chunks per fill (the first captures the jit, untimed)")
  r.add_argument("--device-map", default=None)
  r.add_argument("--out", required=True)
  c = sub.add_parser("compare")
  c.add_argument("a")
  c.add_argument("b")
  args = parser.parse_args()
  (run if args.cmd == "run" else compare)(args)

if __name__ == "__main__": main()
