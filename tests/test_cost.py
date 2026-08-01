"""Cost accounting tests.

These matter more than typical unit tests: the compute cap is the whole basis for
comparing results, so a hole in the counter silently invalidates every experiment.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from harness.cost import UnregisteredOpError, count_cost


class _Conv(nn.Module):
    def __init__(self, kernel_size=6, dilation=41, channels=8):
        super().__init__()
        self.channels = channels
        self.conv = nn.Conv1d(channels, channels, kernel_size, dilation=dilation)

    def forward(self, x):
        return self.conv(x.unsqueeze(1).expand(-1, self.channels, -1))


def test_conv_macs_match_hand_calculation():
    """MACs per output sample = out_channels * in_channels * kernel_size."""
    report = count_cost(_Conv(kernel_size=6, channels=8), input_samples=8192)
    assert report.macs_per_sample == pytest.approx(8 * 8 * 6)


def test_functional_conv_is_counted():
    """A model bypassing nn.Module cannot report zero cost.

    Module hooks would miss this entirely, which is why dispatch is used.
    """

    class Functional(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.randn(8, 8, 6))

        def forward(self, x):
            return F.conv1d(x.unsqueeze(1).expand(-1, 8, -1), self.w)

    report = count_cost(Functional(), input_samples=8192)
    assert report.macs_per_sample == pytest.approx(8 * 8 * 6)


def test_elementwise_tracked_separately_and_costs_no_macs():
    class Elementwise(nn.Module):
        def forward(self, x):
            return torch.tanh(x) * torch.sigmoid(x)

    report = count_cost(Elementwise(), input_samples=8192)
    assert report.macs_per_sample == 0
    assert report.elementwise_per_sample == pytest.approx(3.0)  # tanh, sigmoid, mul


def test_unregistered_op_raises_rather_than_counting_free():
    """Free-by-default would make the cap forgeable."""

    class Fft(nn.Module):
        def forward(self, x):
            return torch.fft.rfft(x).abs()[..., : x.shape[-1] // 2]

    with pytest.raises(UnregisteredOpError):
        count_cost(Fft(), input_samples=8192)


def test_superlinear_cost_is_flagged():
    """Attention over an uncached context has no fixed per-sample cost."""

    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q, self.k, self.v = (nn.Linear(1, 8) for _ in range(3))
            self.o = nn.Linear(8, 1)

        def forward(self, x):
            z = x.unsqueeze(-1)
            a = torch.softmax(self.q(z) @ self.k(z).transpose(-1, -2) / 8**0.5, dim=-1)
            return self.o(a @ self.v(z)).squeeze(-1)

    report = count_cost(Attention(), input_samples=2048)
    assert not report.is_linear


def test_linear_model_is_not_flagged():
    report = count_cost(_Conv(), input_samples=8192)
    assert report.is_linear
