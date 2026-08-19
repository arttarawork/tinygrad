#!/usr/bin/env python3
"""Standalone repro harness for a suspected upstream bug (see docs/rand_fusion_bug.md):
Tensor.rand_like fused into a Gumbel-argmax sampling chain, under a symbolic-shaped TinyJit
graph with a REALIZED scalar `temperature` input, was reported to sometimes emit a wrong
(non-greedy) token at temperature=0 in tinygrad/llm/model.py's Transformer.forward/generate.

This mirrors the exact trigger shape: symbolic-length upstream compute (UOp.variable-bound
slice) -> concrete-shape logits -> Tensor.rand_like(logits, contiguous=False) (a fusion
candidate) -> Gumbel-max argmax, replayed across many bound shapes through one TinyJit capture.

Usage:
  python extra/rand_fusion_bug_repro.py            # fused rand (candidate for the bug)
  python extra/rand_fusion_bug_repro.py --control   # same, but u.realize()'d before use (per
                                                      # the report, this is the known-good path)

Status at af2a43c85 (this repo's baseline): both modes print 0 mismatches across hundreds of
calls on METAL and CPU. See docs/rand_fusion_bug.md for the full investigation, including a
ground-truth threefry differential check (test/unit/test_rand_fusion_bug.py) that is a much
more sensitive probe than this argmax-level check.
"""
import sys
from tinygrad import Tensor, UOp, TinyJit

VOCAB, DIM, MAXLEN = 32, 16, 128
CONTROL = "--control" in sys.argv

Tensor.manual_seed(0)
W = Tensor.randn(DIM, VOCAB).realize()
full = Tensor.randn(1, MAXLEN, DIM).realize()

def forward(x: Tensor, temperature: Tensor) -> Tensor:
  logits = (x[:, -1:] @ W)[:, -1, :]                     # logits depend on symbolic-length x
  u = Tensor.rand_like(logits, contiguous=False)          # candidate for fusion into the argmax kernel
  if CONTROL: u = u.realize()                             # breaks fusion: reported to fix the bug
  gumbel = (u.maximum(1e-12).log().neg()).log()
  return (logits / temperature.maximum(1e-12) - gumbel).argmax(-1, keepdim=True)

jit_forward = TinyJit(forward)
temp = Tensor([0.0])                                      # realized JIT input (forced by _prepare_jit_inputs)
v_len = UOp.variable("T", 1, MAXLEN)

mismatches = 0
for T in [3, 7, 12, 5, 20, 40, 8, 63, 100, 15] * 3:
  vt = v_len.bind(T)
  got = jit_forward(full[:, :vt], temp).item()
  want = int((full[:, T - 1:T] @ W)[:, -1, :].argmax(-1).item())   # eager greedy, no RNG involved
  if got != want:
    mismatches += 1
    print(f"MISMATCH T={T}: jit_fused={got} greedy={want}")

print(f"{'CONTROL' if CONTROL else 'FUSED'}: {mismatches} mismatches")
