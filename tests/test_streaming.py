"""Streaming cost-form tests.

The streaming form is the one place a model gets to say "cost me on something other
than what I ran". That is a hole in the compute cap by construction, so most of what
is here is adversarial: forms that lie, forms that smuggle capacity, forms that are
cheap only over the probe window. Each must be refused rather than believed.
"""

import pytest
import torch
import torch.nn as nn

from harness.cost import count_cost
from harness.streaming import (
    MAX_EQUIVALENCE_ESR,
    MIN_PROBE_SAMPLES,
    StreamingFormError,
)


class _Recurrence(nn.Module):
    """y[t] = a*y[t-1] + w.x[t], as a scan and as a loop.

    Stands in for the whole family: a cheap recurrence trained through an expensive
    parallel form.
    """

    def __init__(self, width=8, chunk=64):
        super().__init__()
        self.w = nn.Parameter(torch.randn(width) / width**0.5)
        self.decay = nn.Parameter(torch.full((width,), 0.9))
        self.out = nn.Linear(width, 1, bias=False)
        self.receptive_field = 1
        self.chunk = chunk

    def _drive(self, x):
        return x.unsqueeze(-1) * self.w  # (B, T, width)

    def _read(self, h):
        return self.out(h).squeeze(-1)

    def forward(self, x):
        """Training form: materialize the intra-chunk impulse response as a GEMM."""
        u = self._drive(x)
        c = min(self.chunk, u.shape[1])
        j = torch.arange(c, dtype=x.dtype)
        lag = (j[:, None] - j[None, :]).clamp(min=0)
        kernel = torch.where(
            j[:, None] >= j[None, :], self.decay[:, None, None] ** lag, torch.zeros(())
        )  # (width, c, c)

        state = u.new_zeros(u.shape[0], u.shape[-1])
        pieces = []
        for start in range(0, u.shape[1], c):
            block = u[:, start : start + c, :]
            n = block.shape[1]
            h = torch.einsum("wtk,bkw->btw", kernel[:, :n, :n], block)
            propagate = self.decay[None, :] ** (j[:n, None] + 1)  # (n, width)
            h = h + propagate[None] * state[:, None, :]
            state = h[:, -1, :]
            pieces.append(h)
        return self._read(torch.cat(pieces, dim=1))

    def _recurrent(self, x):
        u = self._drive(x)
        h = u.new_zeros(u.shape[0], u.shape[-1])
        steps = []
        for t in range(u.shape[1]):
            h = h * self.decay + u[:, t, :]
            steps.append(h)
        return self._read(torch.stack(steps, dim=1))

    def streaming_form(self):
        return _View(self)


class _View(nn.Module):
    cost_probe_samples = 256
    cost_probe_stride = 64

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.receptive_field = inner.receptive_field

    def forward(self, x):
        return self.inner._recurrent(x)


class _ScanOnly(_Recurrence):
    """Same maths, no streaming form offered. The comparison point."""

    streaming_form = None


def test_streaming_form_is_used_and_is_verified():
    torch.manual_seed(0)
    report = count_cost(_Recurrence(), verify_at=2048)

    assert report.used_streaming_form
    assert report.equivalence_esr < MAX_EQUIVALENCE_ESR
    assert report.is_linear
    # width-8 recurrence: a drive, a decay, a readout. Tens of MACs, not thousands.
    assert report.macs_per_sample < 200


def test_costing_the_training_form_instead_would_reject_the_architecture():
    """Why the machinery has to exist at all.

    The scan is O(chunk) per sample. Charging that would put a model that deploys
    for tens of MACs somewhere near A2's entire budget -- not because it is
    expensive, but because of how it was trained.
    """
    torch.manual_seed(0)
    streamed = count_cost(_Recurrence(), verify_at=2048)
    as_trained = count_cost(_ScanOnly(), input_samples=2048)

    assert not as_trained.used_streaming_form
    assert as_trained.macs_per_sample > 20 * streamed.macs_per_sample


def test_disagreeing_streaming_form_is_refused():
    """A cheap form that is not the same function is the obvious attack."""

    class Liar(_Recurrence):
        def streaming_form(self):
            view = _View(self)
            view.forward = lambda x: torch.zeros_like(x)  # maximally cheap
            return view

    torch.manual_seed(0)
    with pytest.raises(StreamingFormError, match="disagrees with the training form"):
        count_cost(Liar())


def test_subtle_disagreement_is_also_refused():
    """Not just zeros: a form that drops one term still fails, well below any ESR
    a run could plausibly claim as a win."""

    class Sloppy(_Recurrence):
        def streaming_form(self):
            view = _View(self)
            inner = self

            def fwd(x):
                y = inner._recurrent(x)
                return y * 1.001  # 0.1% gain error

            view.forward = fwd
            return view

    torch.manual_seed(0)
    with pytest.raises(StreamingFormError, match="disagrees"):
        count_cost(Sloppy())


def test_streaming_form_may_not_add_parameters():
    class Fatter(_Recurrence):
        def streaming_form(self):
            view = _View(self)
            view.extra = nn.Parameter(torch.zeros(1000))
            return view

    torch.manual_seed(0)
    with pytest.raises(StreamingFormError, match="may not add capacity"):
        count_cost(Fatter())


def test_probe_floors_are_enforced():
    class Tiny(_Recurrence):
        def streaming_form(self):
            view = _View(self)
            view.cost_probe_samples = MIN_PROBE_SAMPLES - 1
            return view

    torch.manual_seed(0)
    with pytest.raises(StreamingFormError, match="below the floor"):
        count_cost(Tiny())


def test_cost_that_is_cheap_only_over_the_probe_window_is_caught():
    """The residual risk the extrapolation check exists to close.

    This model is exactly affine over any short probe and does a burst of extra work
    once the input is long -- so the short-probe fit is honest locally and wrong at
    the length that matters.
    """

    class Sneaky(_Recurrence):
        def _recurrent(self, x):
            y = super()._recurrent(x)
            if x.shape[-1] > 1024:
                # Real ops, invisible to a 256/320/384-sample probe, and multiplied
                # out of the result so the equivalence check cannot see them either.
                # Only the cost measurement can catch this.
                junk = torch.einsum("btw,bkw->btk", self._drive(x), self._drive(x))
                y = y + 0.0 * junk.sum(dim=-1)
            return y

    torch.manual_seed(0)
    with pytest.raises(StreamingFormError, match="not affine in input length"):
        count_cost(Sneaky(), verify_at=4096)


def test_models_without_a_streaming_form_are_untouched():
    """The common path must not have changed."""
    from harness.reference import A2_STANDARD_MACS, a2_standard

    report = count_cost(a2_standard(), input_samples=32768)
    assert not report.used_streaming_form
    assert report.equivalence_esr == 0.0
    assert report.macs_per_sample == pytest.approx(A2_STANDARD_MACS, abs=0.5)
