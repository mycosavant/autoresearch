"""Alignment tests.

Dry/wet misalignment inflates ESR by orders of magnitude and looks exactly like a
bad architecture, so a wrong delay estimate can burn an entire night of search.
"""

import numpy as np
import pytest

from prepare import estimate_delay


def _colored_noise(n, seed=0, pole=0.9):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    b = 0.0
    for i in range(n):
        b = pole * b + (1 - pole) * x[i]
        x[i] = b
    return x


@pytest.mark.parametrize("true_delay", [0, 1, 7, 64, 999, 3000])
def test_delay_estimate_is_sample_exact(true_delay):
    """Must survive heavy distortion and a large gain difference.

    A high-gain capture can be tens of dB louder than its input and is severely
    waveshaped, so the estimator has to key on shape rather than amplitude.
    """
    n = 48000 * 25
    x = _colored_noise(n)
    y = np.zeros_like(x)
    y[true_delay:] = np.tanh(8 * x[: n - true_delay])
    y *= 25.0

    assert estimate_delay(x, y) == true_delay


def test_silent_input_is_rejected():
    n = 48000 * 25
    with pytest.raises(ValueError):
        estimate_delay(np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))


def test_too_short_is_rejected():
    x = _colored_noise(4096)
    with pytest.raises(ValueError):
        estimate_delay(x, x)
