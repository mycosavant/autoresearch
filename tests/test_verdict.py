"""Decision-rule tests: the compute gate and the noise floor."""

import pytest

from harness.cost import CostReport
from harness.verdict import Budget, Verdict, decide, noise_floor


def _cost(macs=1000.0, elementwise=10.0):
    return CostReport(
        macs_per_sample=macs,
        elementwise_per_sample=elementwise,
        params=1000,
        output_samples=1,
        total_macs=1,
        total_elementwise=1,
    )


@pytest.fixture
def budget():
    return Budget.from_reference(_cost())


@pytest.fixture
def floor():
    return noise_floor([0.0100, 0.0104, 0.0101])


def test_noise_floor_is_the_spread():
    assert noise_floor([0.010, 0.012, 0.011]) == pytest.approx(0.002)


def test_noise_floor_needs_repeats():
    with pytest.raises(ValueError):
        noise_floor([0.01])


def test_baseline_is_always_kept(budget, floor):
    assert decide(0.010, None, _cost(), budget, floor).verdict is Verdict.KEEP


def test_clear_improvement_kept(budget, floor):
    assert decide(0.008, 0.010, _cost(), budget, floor).verdict is Verdict.KEEP


def test_improvement_within_noise_is_discarded(budget, floor):
    """The defence the upstream loop lacks: small wins are not evidence."""
    decision = decide(0.00995, 0.010, _cost(), budget, floor)
    assert decision.verdict is Verdict.DISCARD
    assert "noise floor" in decision.reason


def test_regression_discarded(budget, floor):
    assert decide(0.012, 0.010, _cost(), budget, floor).verdict is Verdict.DISCARD


def test_over_mac_budget_is_invalid_not_merely_worse(budget, floor):
    """An over-budget model answers a different question."""
    decision = decide(0.001, 0.010, _cost(macs=1500), budget, floor)
    assert decision.verdict is Verdict.INVALID
    assert "MAC budget" in decision.reason


def test_elementwise_dodge_is_invalid(budget, floor):
    """Compute cannot be moved into pointwise ops to duck the MAC cap."""
    decision = decide(0.001, 0.010, _cost(macs=900, elementwise=100), budget, floor)
    assert decision.verdict is Verdict.INVALID
    assert "elementwise" in decision.reason


def test_holdout_regression_warns_but_does_not_veto(budget, floor):
    """Vetoing on the holdout would turn it into another training signal."""
    decision = decide(
        0.007, 0.010, _cost(), budget, floor,
        candidate_holdout=0.020, incumbent_holdout=0.015,
    )
    assert decision.verdict is Verdict.KEEP
    assert "holdout worsened" in decision.reason
