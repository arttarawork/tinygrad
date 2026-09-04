# T4.66k: is the MTP draft path in this tree numerically the same as an older model.py's, on CPU?
#   git show 6bf1b39c1:tinygrad/llm/model.py > /tmp/model_66c.py
#   CHECK_OOB=1 DEV=CPU PYTHONPATH=. .venv/bin/python extra/t466k_draft_equiv.py /tmp/model_66c.py
# Builds the same tiny GDN+MTP model (same seed) with both modules, runs speculative_generate(k=3) on the same prompts,
# and compares the emitted tokens plus every draft() call's inputs (h, token, position) and output logits.
# Result on 4356682da vs 6bf1b39c1 (T4.66c): tokens equal; positions/tokens equal; logits identical for the first
# iterations, ~1e-5 apart (|logits| ~3e-2) once a partial accept has gone through CHECKPOINT+REDO instead of 66b's
# capture-assign -- fp noise, no mechanism for an acceptance collapse. The tinygrad core is identical between the two
# commits (only tinygrad/llm/model.py differs), so the two modules run on the same kernels.
import importlib.util, sys
from dataclasses import replace
import numpy as np
from tinygrad import Tensor, nn
from tinygrad.helpers import ContextVar
from tinygrad.uop.ops import Ops
import tinygrad.llm.model as head
from test.unit.test_mtp_load import VOCAB, DIM, HIDDEN, N_HEADS, N_KV_HEADS, HEAD_DIM

def load_old(path:str):
  spec = importlib.util.spec_from_file_location("model_old", path)
  assert spec is not None and spec.loader is not None
  mod = importlib.util.module_from_spec(spec)
  sys.modules["model_old"] = mod
  saved = dict(ContextVar._cache)
  ContextVar._cache.clear()  # the old module re-declares GDN_CHUNK/SPEC_* -- let it
  spec.loader.exec_module(mod)
  ContextVar._cache.update(saved)
  return mod

def build(mod, seed=0, max_context=64):
  SSM, TC = getattr(mod, "SSMConfig", head.SSMConfig), getattr(mod, "TransformerConfig", head.TransformerConfig)
  ssm = SSM(conv_kernel=4, state_size=4, group_count=1, time_step_rank=2, inner_size=8)
  cfg = TC(num_blocks=2, dim=DIM, hidden_dim=HIDDEN, n_heads=N_HEADS, n_kv_heads=N_KV_HEADS, norm_eps=1e-5, vocab_size=VOCAB,
           head_dim=HEAD_DIM, rope_theta=10000.0, rope_dim=HEAD_DIM, v_head_dim=HEAD_DIM, max_context=max_context,
           attn_output_gate=True, ssm_layers=(True, False), ssm=ssm)
  m = mod.Transformer(cfg)
  m.mtp_head = mod.MTPHead(replace(cfg, qk_norm=cfg.head_dim), mod.TransformerBlock)
  Tensor.manual_seed(seed)
  params = nn.state.get_parameters(m)
  for p in params: p.replace(Tensor.randn(*p.shape) * 0.1)
  Tensor.realize(*params)
  return m

def run(mod, prompt:list[int], n:int):
  rec: list[tuple] = []
  orig = mod.MTPHead.draft
  def draft(self, owner, h, tok_ids, start_pos):
    out = orig(self, owner, h, tok_ids, start_pos)
    pos = [u.arg for u in start_pos.toposort() if u.op is Ops.CONST][-1] if hasattr(start_pos, "toposort") else start_pos
    rec.append((out.numpy().copy(), h.numpy().copy(), int(tok_ids.item()), pos))
    return out
  mod.MTPHead.draft = draft
  try:
    m = build(mod)
    toks = [v for _, v in zip(range(n), m.speculative_generate(list(prompt), k=3))]
  finally: mod.MTPHead.draft = orig
  return toks, rec

if __name__ == "__main__":
  old = load_old(sys.argv[1])
  for prompt in ([5, 2, 3, 4], [1, 2, 3], [7]):
    ta, ra = run(old, prompt, 10)
    tb, rb = run(head, prompt, 10)
    worst = max(float(np.abs(a[0] - b[0]).max()) for a, b in zip(ra, rb))
    same_io = all(a[2] == b[2] and a[3] == b[3] for a, b in zip(ra, rb))
    print(f"prompt={prompt}: tokens equal={ta == tb} ({ta}); draft calls {len(ra)}/{len(rb)}, same (token, position) inputs={same_io}, "
          f"max|dlogits diff|={worst:.2e}")
    for i, (a, b) in enumerate(zip(ra, rb)):
      print(f"  call {i}: pos {a[3]}/{b[3]} tok {a[2]}/{b[2]} max|dlogits| {float(np.abs(a[0] - b[0]).max()):.2e} "
            f"max|h| {float(np.abs(a[1] - b[1]).max()):.2e}")
