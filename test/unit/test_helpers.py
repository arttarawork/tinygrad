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

class TestTarMembers(unittest.TestCase):
  @staticmethod
  def _tar(*entries):
    import io, tarfile
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
      for name, kind, arg in entries:
        ti = tarfile.TarInfo(name)
        if kind == "file":
          ti.size = len(arg)
          tf.addfile(ti, io.BytesIO(arg))
        else:
          ti.type, ti.linkname = {"dir": tarfile.DIRTYPE, "sym": tarfile.SYMTYPE, "hard": tarfile.LNKTYPE, "fifo": tarfile.FIFOTYPE}[kind], arg or ""
          tf.addfile(ti)
    buf.seek(0)
    return tarfile.open(fileobj=buf)

  def test_benign_archive_with_internal_links(self):
    from tinygrad.helpers import tar_members
    tf = self._tar(("a", "dir", None), ("a/x", "file", b"x"), ("a/l", "sym", "x"), ("lib.so.1", "sym", "a/x"), ("lib.so", "sym", "lib.so.1"),
                   ("h", "hard", "a/x"))
    self.assertEqual(len(tar_members(tf, "/tmp/dest")), 6)

  def test_rejects_escapes(self):
    from tinygrad.helpers import tar_members
    for entries in [[("../evil", "file", b"")], [("/abs", "file", b"")], [("a/../../evil", "file", b"")],
                    [("l", "sym", "../..")], [("l", "sym", "/etc")], [("h", "hard", "../outside")], [("f", "fifo", None)],
                    [("l", "sym", "a"), ("l/x", "file", b"")],               # writes through an earlier link
                    [("l", "sym", "a"), ("L/x", "file", b"")],               # ... spelled in another case (APFS)
                    [("l", "sym", "a"), ("l", "file", b"")],                 # overwrites the link (opening it follows it)
                    [("p", "sym", "."), ("q", "sym", "p/p/..")],             # link chain that escapes once p is resolved
                    [("", "sym", "..")]]:
      with self.assertRaises(ValueError, msg=str(entries)): tar_members(self._tar(*entries), "/tmp/dest")
