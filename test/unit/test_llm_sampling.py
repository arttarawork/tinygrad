import contextlib, itertools, re, sys, unittest
from unittest import mock
from tinygrad import Tensor, nn
from tinygrad.uop.ops import Ops
from tinygrad.llm.model import Transformer, TransformerConfig

def _tiny_model():
  Tensor.manual_seed(42)
  model = Transformer(TransformerConfig(num_blocks=2, dim=32, hidden_dim=64, n_heads=4, n_kv_heads=2, norm_eps=1e-5,
                                        vocab_size=100, head_dim=8, rope_theta=10000.0, rope_dim=8, v_head_dim=8, max_context=32))
  Tensor.realize(*nn.state.get_parameters(model))
  return model

class TestGreedySampling(unittest.TestCase):
  def _forward_linear(self, temperature):
    linear, _ = _tiny_model().forward(Tensor([[1]], dtype="int32"), 0, temperature).linear_with_vars()
    return linear

  def test_greedy_has_no_rng_kernels(self):
    self.assertFalse(any(u.op is Ops.THREEFRY for u in self._forward_linear(None).toposort()))
    self.assertTrue(any(u.op is Ops.THREEFRY for u in self._forward_linear(Tensor([0.7])).toposort()))

  def test_greedy_matches_old_temp0(self):
    # old temp=0 path (logits/1e-12 - gumbel noise) must pick the same token as plain argmax (up to exact-tie logits)
    model = _tiny_model()
    tok = Tensor([[1]], dtype="int32")
    self.assertEqual(model.forward(tok, 0, None).item(), model.forward(tok, 0, Tensor([0.0])).item())

  def test_no_recapture_across_temperature_switch(self):
    model = _tiny_model()
    def take(n, temp): return [t for _, t in zip(range(n), model.generate([1, 2, 3], temperature=temp))]
    a = take(4, 0.0)
    caps = {k: j.captured for k, j in model.jit.items()}
    self.assertIsNotNone(caps[(False, False, None, False)])  # greedy rollout was jitted
    take(4, 0.7)
    b = take(4, 0.0)
    self.assertEqual(a, b)  # greedy is deterministic and unaffected by sampled runs
    # the sampled run captured its own jits and did not recapture the greedy ones
    # (jit entries are now created lazily, T4.12 -- a key absent from `caps` just wasn't hit yet)
    for k, j in model.jit.items():
      if caps.get(k) is not None: self.assertIs(j.captured, caps[k])
    self.assertIsNotNone(model.jit[(False, False, None, False)].captured)

class TestPrefillFamilySampling(unittest.TestCase):
  """T4.139: the pruned prefill family samples every request's first token; its Gumbel draw must advance Tensor.rand's counter on every
  replay -- a pruned draw repeated the capture's noise, so a prompt at temperature 1.0 got the same first token every time."""
  def test_first_token_draw_advances_every_request(self):
    model = _tiny_model()
    dev = model.blk[-1].device
    def first(prompt):
      before = Tensor._device_rng_counters[dev].tolist()[0]
      tok = next(model.generate(prompt, temperature=1.0))  # the prefill chunk's own draw, no decode step
      return tok, Tensor._device_rng_counters[dev].tolist()[0] - before
    got = [first([1, 2, 3]) for _ in range(6)] + [first([4, 5, 6]) for _ in range(2)]
    self.assertEqual([d for _, d in got], [100] * 8)  # one vocab-sized draw per request
    self.assertGreater(len({t for t, _ in got[:6]}), 1, got)

if __name__ == '__main__':
  unittest.main()


class TestMinPSampling(unittest.TestCase):
  """T4.85: min-p drops the tail below min_p x top probability; the top token always survives; min_p=0 leaves the tail reachable."""
  def test_tail_never_sampled_with_min_p(self):
    from tinygrad.llm.model import sample_logits
    logits = Tensor([[6.0, 5.9, -4.0, -4.0]]).expand(64, 4).contiguous()   # two near-ties on top, two tokens ~1e-4 of the top
    ids = sample_logits(logits, Tensor([1.0]), min_p=0.5).numpy().ravel().tolist()
    self.assertTrue(set(ids) <= {0, 1}, ids)
    self.assertEqual(sample_logits(logits, Tensor([1.0]), min_p=1.0).numpy().ravel().tolist(), [0]*64)   # only the top survives
    ids0 = sample_logits(Tensor([[0.0, 0.0, 0.0, 0.0]]).expand(256, 4).contiguous(), Tensor([1.0]), min_p=0.0).numpy().ravel().tolist()
    self.assertEqual(set(ids0), {0, 1, 2, 3})   # uniform logits, no cut: every token reachable


class TestMinPMaskLogDomain(unittest.TestCase):
  """T4.125c: `scaled >= scaled.max() + log(min_p)` (the mask sample_logits uses now) must select exactly the same
  tokens as the softmax-based mask it replaced, `probs >= probs.max() * min_p` (softmax is monotone in `scaled`
  and its normalizer cancels out of the ratio -- these are the same test by construction, not by luck) -- checked
  on random logits that include ties, -inf entries (what a presence penalty or a vocab mask produces) and both
  very small and very large temperatures."""
  def test_new_mask_matches_old_softmax_mask(self):
    import math
    import numpy as np
    rng = np.random.default_rng(125)
    for min_p in (1e-6, 0.1, 0.5, 0.9, 1.0):
      for temperature in (1e-3, 0.6, 1.0, 5.0, 100.0):
        logits_np = rng.normal(size=(16, 37)).astype(np.float32)
        logits_np[:, 2] = -np.inf                 # a masked-out token (presence penalty / vocab mask)
        logits_np[0, 3] = logits_np[0, 4] = 10.0   # an explicit tie at the top of row 0
        logits = Tensor(logits_np)
        temp_t = Tensor([temperature])
        scaled = logits / temp_t.maximum(1e-12)
        probs = scaled.softmax(-1)
        old_mask = (probs >= probs.max(-1, keepdim=True) * min_p).numpy()
        new_mask = (scaled >= scaled.max(-1, keepdim=True) + math.log(min_p)).numpy()
        np.testing.assert_array_equal(old_mask, new_mask, err_msg=f"min_p={min_p} temperature={temperature}")
        # the top token always survives, every row, under both masks
        top = logits_np.argmax(-1)
        self.assertTrue(old_mask[np.arange(16), top].all())
        self.assertTrue(new_mask[np.arange(16), top].all())
        # the tied pair at row 0's top both survive (a tie is not accidentally broken by the log-domain rewrite)
        self.assertTrue(old_mask[0, 3] and old_mask[0, 4] and new_mask[0, 3] and new_mask[0, 4])



class TestNVSampleKernels(unittest.TestCase):
  """T4.146: kernels/nv_sample.py -- the head's pick as head_max_nv / head_draw_nv / head_pick_nv -- EXECUTED on the PYTHON emulator (the
  xor shuffle swapped for a block-shared exchange) against the generic graph forward() builds (argmax; (temp > 0).where(sample_logits,
  argmax)), from the same logits and, after the same Tensor.manual_seed, the same Tensor.rand_like uniforms: the ids must be EQUAL, so
  the distribution is the generic one. Ties at the top, -inf entries, a vocabulary that is not a multiple of a block (2048) and one that
  is, the head gemv's split-K partials or plain logits, two rows, temperatures 0 / 0.7 / 1.3, min_p 0 / 0.05 / 0.5, several seeds."""
  _slot = itertools.count(400)

  @classmethod
  def _emulated_xor(cls, val, off:int):
    from tinygrad import UOp, dtypes
    from tinygrad.dtype import AddrSpace
    from tinygrad.uop.ops import AxisType
    from tinygrad.llm.kernels import nv_sample
    if not any(u.op is Ops.RANGE and u.arg[-1] is AxisType.WARP for u in val.toposort()): return val  # the same in every lane
    lane, warp = UOp.range(32, -1, AxisType.WARP), UOp.range(nv_sample.WARPS, 1, AxisType.LOCAL)
    tid = warp * 32 + lane
    buf = UOp.placeholder((nv_sample.THREADS,), dtypes.float32, slot=next(cls._slot), addrspace=AddrSpace.LOCAL)
    return buf.after(buf[tid].store(val))[tid + off - (tid // off) % 2 * (2 * off)].load()  # tid ^ off, z3-boundable

  @contextlib.contextmanager
  def _emulated(self):
    from tinygrad.llm.kernels import nv_sample
    with contextlib.ExitStack() as stack:
      stack.enter_context(mock.patch.object(nv_sample, "_xor", self._emulated_xor))
      for n in ("_head_max_kernel", "_head_draw_kernel", "_head_pick_kernel"):  # uncached: the emulated shuffle built in
        stack.enter_context(mock.patch.object(nv_sample, n, getattr(nv_sample, n).__wrapped__))
      yield

  @staticmethod
  def _logits(rows:int, vocab:int, C:int, seed:int):
    import numpy as np
    rng = np.random.default_rng(seed)
    parts = (rng.standard_normal((rows, vocab, max(C, 1))) * 1.5).astype(np.float32)
    parts[0, [7, vocab - 3], :] = parts[0].max() + 1  # a tie at the top of row 0: the first index wins
    parts[:, 11, 0] = -np.inf  # -inf entries (a presence penalty / vocab mask shape)
    return parts if C else parts[..., 0]

  @unittest.skipIf(sys.version_info < (3, 12), "the PYTHON emulator")
  def test_ids_equal_the_generic_pick(self):
    from tinygrad.llm.model import sample_logits
    from tinygrad.llm.kernels.nv_sample import nv_sample
    for (rows, vocab, C), seed in itertools.product(((2, 3000, 3), (1, 4096, 0)), range(3)):
      src = Tensor(self._logits(rows, vocab, C, seed), device="PYTHON")
      logits = src.sum(-1) if C else src
      for temp, min_p in ((None, 0.0), (0.0, 0.05), (0.7, 0.0), (0.7, 0.05), (1.3, 0.5)):
        with self.subTest(rows=rows, vocab=vocab, C=C, seed=seed, temp=temp, min_p=min_p):
          t = None if temp is None else Tensor([temp], device="PYTHON")
          Tensor.manual_seed(100 + seed)
          want = (logits.argmax(-1, keepdim=True) if t is None else
                  (t > 0).reshape(1, 1).where(sample_logits(logits, t, min_p), logits.argmax(-1, keepdim=True))).numpy()
          Tensor.manual_seed(100 + seed)
          with self._emulated(): got = nv_sample(logits, t, min_p, src if C else None).numpy()
          self.assertEqual(got.dtype, want.dtype)
          self.assertEqual(got.tolist(), want.tolist())
          if temp is None: self.assertEqual(got[0, 0], 7)  # the tie at the top of row 0

  def test_rng_counter_advance_matches_sample_logits(self):
    # T4.139's counter logic: one vocab-sized Tensor.rand_like per draw, as sample_logits (graphs built, never run)
    from tinygrad.llm.model import sample_logits
    from tinygrad.llm.kernels.nv_sample import nv_sample
    logits = Tensor.empty(1, 4096)
    def advance(fn) -> int:
      Tensor.manual_seed(5)
      fn(logits, Tensor([0.7]), 0.05)
      return Tensor._device_rng_counters[logits.device].tolist()[0]
    self.assertEqual(advance(nv_sample), advance(sample_logits))

  def test_render_at_27b(self):
    # the 27B head (248320 logits, the Q6_K gemv's 5 split-K partials) on sm_86 source: 122 blocks of 256 threads for the sum + block argmax
    # and for the draw, one 256-thread block for the pick; greedy-only: no draw kernel and no uniforms
    from tinygrad.llm.kernels.nv_sample import nv_sample
    from test.unit.test_llm_nv_scan import _render_cuda
    from test.unit.test_llm_nv_gdn_prep import TestNvrtcRegisters
    for temp in (Tensor.empty(1), None):
      tok = nv_sample(Tensor.empty(1, 248320), temp, 0.05, Tensor.empty(1, 248320, 5))
      calls = [u.src[0] for u in tok.uop.toposort() if u.op is Ops.CALL and u.src[0].arg.name.startswith("head_")]
      self.assertEqual([c.arg.name for c in calls], ["head_max_nv", "head_draw_nv", "head_pick_nv"][::1 if temp is not None else 2])
      self.assertEqual(any(u.op is Ops.THREEFRY for u in tok.uop.toposort()), temp is not None)
      for sink in calls:
        with self.subTest(kernel=sink.arg.name, sampled=temp is not None):
          src = _render_cuda(sink)
          self.assertIn("__launch_bounds__(256)", src)
          self.assertEqual(re.findall(r"blockIdx\.[xyz]; /\* (\d+) \*/", src), [] if sink.arg.name == "head_pick_nv" else ["122"])
          try: regs, stack = TestNvrtcRegisters._attrs(src)  # docker nvrtc lane; 11-01: 40 / 45 / 35 registers, no stack
          except Exception as e: self.skipTest(f"nvrtc lane unavailable: {e!r}")
          self.assertEqual(stack, 0)

  def test_forward_takes_the_kernels_under_the_gate(self):
    # model.py's wiring, scheduled only (the gate mocked open on CPU): the pick is the head_* kernels -- no generic vocab-wide argmax reduce
    # left -- and the sampled family still draws its uniforms (THREEFRY); a presence penalty keeps the generic graph
    from tinygrad.uop.ops import KernelInfo
    def names(temperature, penalty=None) -> list[str]:
      model = _tiny_model()
      if penalty is not None: model.penalty_mask = Tensor.zeros(1, 100).contiguous().realize()
      with mock.patch("tinygrad.llm.model.nv_sample_ok", return_value=True):
        linear, _ = model.forward(Tensor([[1]], dtype="int32"), 0, temperature, presence_penalty=penalty).linear_with_vars()
      calls = [u.src[0] for u in linear.src if u.op is Ops.CALL]
      return [c.arg.name for c in calls if isinstance(c.arg, KernelInfo) and c.arg.name.startswith("head_")], \
             any(u.op is Ops.THREEFRY for u in linear.toposort())
    self.assertEqual(names(Tensor([0.7])), (["head_max_nv", "head_draw_nv", "head_pick_nv"], True))
    self.assertEqual(names(None), (["head_max_nv", "head_pick_nv"], False))
    self.assertEqual(names(Tensor([0.7]), Tensor([1.5]))[0], [])
