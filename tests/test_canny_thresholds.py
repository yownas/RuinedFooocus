import math
import os
import sys
import unittest

# Ensure project root is importable when running this file directly.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.canny_utils import sanitize_canny_thresholds


class TestCannyThresholds(unittest.TestCase):
    def test_zero_low_threshold_is_clamped_to_positive(self):
        low, high = sanitize_canny_thresholds(0.0, 0.5)
        self.assertGreater(low, 0.0)
        self.assertLess(low, high)
        self.assertLessEqual(high, 1.0)
        self.assertTrue(math.isclose(high, 0.5))

    def test_negative_thresholds_are_clamped_and_ordered(self):
        low, high = sanitize_canny_thresholds(-1.0, 2.0)
        self.assertGreater(low, 0.0)
        self.assertLess(low, high)
        self.assertLess(high, 1.0)

    def test_high_less_than_low_is_corrected(self):
        low, high = sanitize_canny_thresholds(0.5, 0.1)
        self.assertGreater(low, 0.0)
        self.assertLess(low, high)
        self.assertLess(high, 1.0)


if __name__ == "__main__":
    unittest.main()
