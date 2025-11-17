import math
from typing import Tuple


def sanitize_canny_thresholds(low: float, high: float) -> Tuple[float, float]:
    """
    Normalize Canny thresholds to the valid range expected by kornia.

    kornia.filters.canny requires 0 < low, high < 1 and low < high.
    Users can currently set the sliders to 0.0 (or other invalid values),
    which would cause a runtime error. This helper clamps values into a
    safe range and enforces ordering while preserving the intent as much
    as possible.
    """
    eps = 1e-6
    max_val = 1.0 - eps

    # Handle NaNs / infinities by falling back to reasonable defaults.
    if not math.isfinite(low):
        low = eps
    if not math.isfinite(high):
        high = max_val

    # Clamp into the open interval (0, 1).
    if low <= 0.0:
        low = eps
    if high <= 0.0:
        high = max_val

    low = max(eps, min(low, max_val))
    high = max(eps, min(high, max_val))

    # Ensure low < high. When they are equal or inverted, keep the
    # lower bound as-is and nudge the upper bound upwards.
    if not low < high:
        # If low is already near the maximum, move it slightly down
        # so that we still have a valid open interval.
        if low >= max_val:
            low = max_val - eps
            high = max_val
        else:
            high = min(max_val, low + eps)

    return float(low), float(high)

