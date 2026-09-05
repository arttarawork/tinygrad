# T4.66i: does warmup() (with/without the 66g delattr) change the MAIN model's speculative output vs a fresh model?
# Matrix: {single device, CPU:0/CPU:1 split} x {prompt starts with 0, doesn't} x {fresh, warmup+delattr(66g), warmup-no-delattr, warmup+delattr+reset _cached_tokens}
import sys, os
sys.path.insert(0, 'test/unit')
from dataclasses import replace
from tinygrad import Tensor, nn
from tinygrad.llm.model import Transformer, MTPHead, TransformerBlock
import test_spec_decode as T

def fresh(split:bool, seed:int=0) -> Transformer:
  ref = T._load_gdn(seed=seed, max_context=64)
  if not split: return ref
  cfg = ref.blk[0].config
  m = Transformer(cfg, device_map="CPU:0,CPU:1")
  m.mtp_head = MTPHead(replace(cfg, qk_norm=cfg.head_dim), TransformerBlock)
  for p in nn.state.get_parameters(m.mtp_head): p.to_(m.blk[-1].device)
  nn.state.load_state_dict(m, nn.state.get_state_dict(ref), verbose=False, realize=False)
  m.realize_placement()
  return m

def warm_no_delattr(model):
  for temperature in (0.0, 1.0):
    for _ in range(2): list(zip(range(2), model.generate([0], temperature=temperature)))
  for _ in range(2): list(zip(range(2), model.speculative_generate([0])))

def run(model, prompt, n=8): return [v for _, v in zip(range(n), model.speculative_generate(list(prompt), k=3))]

for split in (False, True):
  for prompt in ([0, 2, 3, 4], [5, 2, 3, 4]):
    ref = [v for _, v in zip(range(8), fresh(split).generate(list(prompt), temperature=0.0))]
    a = run(fresh(split), prompt)
    m = fresh(split); m.warmup(); b = run(m, prompt)                       # 66g warmup (with delattr)
    m = fresh(split); warm_no_delattr(m); c = run(m, prompt)               # warmup minus delattr (66f behavior)
    m = fresh(split); m.warmup(); m._cached_tokens = []; d = run(m, prompt)  # 66g + cached_tokens reset
    tag = f"split={split} prompt0={prompt[0]}"
    print(f"{tag}: ref={ref}")
    print(f"  fresh_spec      {'OK ' if a==ref else 'DIFF'} {a}")
    print(f"  warmup66g       {'OK ' if b==ref else 'DIFF'} {b}")
    print(f"  warmup_noDel    {'OK ' if c==ref else 'DIFF'} {c}")
    print(f"  warmup66g+reset {'OK ' if d==ref else 'DIFF'} {d}")
