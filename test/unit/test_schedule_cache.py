import unittest
import functools
from tinygrad import Tensor, Variable, UOp, function
from tinygrad.uop.ops import KernelInfo
from tinygrad.helpers import Context
from tinygrad.schedule import schedule_cache

def custom_add_kernel(A:UOp, B:UOp, num:int=0) -> UOp:
  return A[0].set(B[0] + num).sink(arg=KernelInfo(f"custom_add_{num}"))

def custom_add_backward(grad_output:UOp, _) -> tuple[None, UOp]:
  grad = Tensor.invalids(*grad_output.shape, dtype=grad_output.dtype, device=grad_output.device)
  grad = Tensor.custom_kernel(grad, Tensor(grad_output, device=grad_output.device), fxn=functools.partial(custom_add_kernel, num=0))[0]
  return None, grad.uop

class TestScheduleCache(unittest.TestCase):
  def test_bound_variable_reuses_cache(self):
    schedule_cache.clear()
    v = Variable('v', 1, 100)
    x = Tensor.ones(10).contiguous().realize()

    # first run with v=5
    t1 = (x + Tensor(v.bind(5))).sum()
    self.assertEqual(t1.item(), 60.0)
    cache_size_after_first = len(schedule_cache)

    # second run with v=10 should reuse cache
    t2 = (x + Tensor(v.bind(10))).sum()
    self.assertEqual(t2.item(), 110.0)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_pcontig_not_cross_served(self):
    # T4.9: SCACHE used to key on structural graph hash alone, so scheduling the *same* tensor
    # expression under two different PCONTIG settings in one process would serve the first PCONTIG
    # value's cached (fused-or-not) schedule to the second call, silently making PCONTIG-vs-baseline
    # comparisons vacuous (see PCONTIG_ATTN_NOTES.md). Build a fresh, identical (structurally) tensor
    # graph for each setting -- rebuilt each time so neither call mutates a shared tensor's .uop.
    schedule_cache.clear()
    def build() -> Tensor:
      a, c = Tensor.empty(4, 2, 16, 8), Tensor.empty(4, 2, 8, 8)
      return a.softmax(-1) @ c

    with Context(PCONTIG=0): linear_off, _ = build().linear_with_vars()
    with Context(PCONTIG=2): linear_on, _ = build().linear_with_vars()

    # each PCONTIG setting must land in its own cache slot, never share one
    self.assertEqual(len(schedule_cache), 2)
    # PCONTIG=2 fuses this softmax@matmul shape into fewer kernels than the PCONTIG=0 baseline;
    # if the cache had cross-served, these would be equal (both from whichever ran first)
    self.assertNotEqual(len(linear_off.src), len(linear_on.src))

  def test_custom_kernel(self):
    for i in range(4):
      a, b = Tensor.empty(1), Tensor.ones(1)
      a = Tensor.custom_kernel(a, b, fxn=functools.partial(custom_add_kernel, num=i))[0]
      a.realize()
      self.assertEqual(a.item(), i+1)

  def test_same_custom_function_reuses_cache(self):
    schedule_cache.clear()
    fxn = functools.partial(custom_add_kernel, num=10)

    # first run
    a, x = Tensor.empty(1), Tensor.ones(1)
    a = Tensor.custom_kernel(a, x, fxn=fxn)[0]
    a.realize()
    self.assertEqual(a.item(), 11)
    cache_size_after_first = len(schedule_cache)

    # second run with same function should reuse cache
    b, x = Tensor.empty(1), Tensor.ones(1)
    b = Tensor.custom_kernel(b, x, fxn=fxn)[0]
    b.realize()
    self.assertEqual(b.item(), 11)
    self.assertEqual(len(schedule_cache), cache_size_after_first)

  def test_simple(self):
    a = Tensor.ones(10).contiguous()
    b = Tensor.ones(10).contiguous()
    Tensor.realize(a, b)

    # warm up
    for _ in range(2):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)

    # confirm schedule cache doesn't grow
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      num = (a.sum().contiguous()+b.sum().contiguous()).item()
      print(num)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)

  def test_simple_precompile(self):
    @function(precompile=True, precompile_backward=True)
    def f(x:Tensor) -> Tensor:
      out = Tensor.invalids(*x.shape, dtype=x.dtype, device=x.device)
      out = Tensor.custom_kernel(out, x, fxn=functools.partial(custom_add_kernel, num=10), grad_fxn=custom_add_backward)[0]
      return out + x

    # warmup
    x = Tensor.ones(1).realize()
    out = f(x)
    out.backward(x)
    self.assertEqual(out.item(), 12)
    self.assertEqual(x.grad.item(), 2)

    # use the cache next time function is called
    start_len_schedule_cache = len(schedule_cache)
    for _ in range(3):
      x = Tensor.ones(1).realize()
      out = f(x)
      out.backward(x)
      self.assertEqual(out.item(), 12)
      self.assertEqual(x.grad.item(), 2)
    self.assertEqual(len(schedule_cache), start_len_schedule_cache)

if __name__ == "__main__":
  unittest.main()
