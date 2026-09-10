import unittest
from extra.launch_floor import measure

class TestLaunchFloor(unittest.TestCase):
  def test_ten_kernels_reports_ten_and_positive_time(self):
    ms_per_replay, us_per_launch, kernels_per_replay = measure(10, 3)
    self.assertEqual(kernels_per_replay, 10)
    self.assertGreater(ms_per_replay, 0)
    self.assertGreater(us_per_launch, 0)

if __name__ == "__main__":
  unittest.main()
