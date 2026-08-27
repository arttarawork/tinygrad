import unittest, gc
from unittest.mock import patch
import numpy as np
from tinygrad.helpers import polyN, disable_gc, wait_cond
from tinygrad.tensor import Tensor, is_numpy_ndarray

class TestPolyN(unittest.TestCase):
  def test_tensor(self):
    np.testing.assert_allclose(polyN(Tensor([1.0, 2.0, 3.0, 4.0]), [1.0, -2.0, 1.0]).numpy(), [0.0, 1.0, 4.0, 9.0])

class TestIsNumpyNdarray(unittest.TestCase):
  def test_tensor_numpy(self):
    self.assertTrue(is_numpy_ndarray(Tensor([1, 2, 3]).numpy()))

class TestDisableGC(unittest.TestCase):
  def test_recursive_decorator(self):
    was_enabled = gc.isenabled()
    @disable_gc()
    def recurse(depth:int):
      self.assertFalse(gc.isenabled())
      if depth: recurse(depth-1)
      self.assertFalse(gc.isenabled())
    try:
      recurse(2)
      self.assertEqual(gc.isenabled(), was_enabled)
    finally:
      (gc.enable if was_enabled else gc.disable)()

class TestWaitCond(unittest.TestCase):
  def test_returns_on_first_match(self):
    self.assertEqual(wait_cond(lambda: True, value=True, timeout_ms=1000), True)

  def test_normal_timeout_raises_timeout_error_with_last_value(self):
    # negative control: the ordinary (deadline-in-the-future, condition-never-met) path must still
    # raise TimeoutError -- with the same message shape -- not regress into any other exception.
    with self.assertRaisesRegex(TimeoutError, r"condition not met: False != True"):
      wait_cond(lambda: False, value=True, timeout_ms=20)

  def test_deadline_already_elapsed_raises_timeout_not_unbound_local(self):
    # T4.45 (found by T4.40c): if the deadline has already passed by the time the loop's first
    # condition check runs, wait_cond must still raise TimeoutError, not UnboundLocalError from
    # referencing a `val` that a pre-fix `while <time check>:` loop guard never let get assigned.
    # A clock that jumps straight past the deadline on its second read reproduces that: one read for
    # start_time, one read for the (pre-fix: loop-guard / post-fix: post-call) elapsed check.
    with patch("tinygrad.helpers.time.perf_counter", side_effect=[0.0, 1000.0]):
      with self.assertRaises(TimeoutError):
        wait_cond(lambda: False, value=True, timeout_ms=10)

if __name__ == '__main__':
  unittest.main()
