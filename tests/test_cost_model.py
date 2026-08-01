"""Cost-model tests.

The point of this model is to be honest about what it does and does not know, so
most of these test the identifiability reporting rather than the arithmetic.
"""

import pytest

from harness.cost import CostReport
from harness.cost_model import (
    DEFAULT_MODEL,
    Measurement,
    available_profiles,
    features_of,
    fit,
    load_profile,
)

DEGENERATE = load_profile("modulus-i9-14900f")
CALIBRATED = load_profile("devcontainer-4cpu")
from harness.verdict import Budget, check_budget


def _report(macs=11776.0, elementwise=729.0, conv_ops=71):
    return CostReport(
        macs_per_sample=macs,
        elementwise_per_sample=elementwise,
        params=12145,
        output_samples=1,
        total_macs=1,
        total_elementwise=1,
        op_counts={"convolution": conv_ops},
    )


def test_profiles_are_discoverable():
    assert {"devcontainer-4cpu", "modulus-i9-14900f"} <= set(available_profiles())


def test_unknown_profile_raises_with_guidance():
    with pytest.raises(FileNotFoundError, match="tools/calibrate.py"):
        load_profile("no-such-machine")


def test_degenerate_profile_knows_it_is_not_adequate():
    """Two structurally identical points cannot identify a per-layer term."""
    assert not DEGENERATE.is_adequate
    assert "conv_ops" not in DEGENERATE.identifiable
    assert any("constant across all measurements" in n for n in DEGENERATE.notes)


def test_degenerate_profile_still_reproduces_published_realtime_factors():
    """modulus published 6.6x (Full) and 24.2x (Lite). Interpolation, not evidence."""
    xrt = lambda f: 1e6 / 48000.0 / DEGENERATE.predict(f)  # noqa: E731
    full = {"macs": 11776.0, "elementwise": 729.0, "conv_ops": 71.0, "const": 1.0}
    lite = {"macs": 1731.0, "elementwise": 274.0, "conv_ops": 71.0, "const": 1.0}
    assert xrt(full) == pytest.approx(6.6, abs=0.1)
    assert xrt(lite) == pytest.approx(24.2, abs=0.1)


def test_measured_profile_is_adequate_and_identified():
    """Eight architectures spanning depth and width identify every coefficient."""
    assert CALIBRATED.is_adequate
    assert CALIBRATED.identifiable == {"macs", "elementwise", "conv_ops", "const"}
    # Residual should sit within the measurement spread (3-9% observed).
    assert CALIBRATED.residual_rel_error < 0.10


def test_measured_profile_charges_for_depth():
    """The whole point: equal MACs, different depth, materially different cost.

    On the measured profile a 26 -> 140 convolution increase at fixed arithmetic
    costs ~1.49x. A MAC-only model scores these two identically, which is the bias
    this whole module exists to remove.
    """
    shallow = {"macs": 5000.0, "elementwise": 300.0, "conv_ops": 26.0, "const": 1.0}
    deep = {"macs": 5000.0, "elementwise": 300.0, "conv_ops": 140.0, "const": 1.0}
    assert CALIBRATED.predict(deep) / CALIBRATED.predict(shallow) == pytest.approx(1.49, abs=0.1)


def test_unset_profile_is_uncalibrated_and_does_not_gate():
    """Default ships uncalibrated: another machine's coefficients are not applied."""
    assert not DEFAULT_MODEL.is_adequate
    assert DEFAULT_MODEL.n_measurements == 0
    assert any("No calibration profile selected" in n for n in DEFAULT_MODEL.notes)


def test_constant_feature_is_reported_unidentifiable():
    measurements = [
        Measurement("a", {"macs": 100.0, "elementwise": 10.0, "conv_ops": 5.0, "const": 1.0}, 1.0),
        Measurement("b", {"macs": 200.0, "elementwise": 20.0, "conv_ops": 5.0, "const": 1.0}, 2.0),
    ]
    model = fit(measurements)
    assert "conv_ops" not in model.identifiable


def test_varying_depth_makes_conv_ops_identifiable():
    """The fix for the shipped calibration: measure at differing depths."""
    truth = {"macs": 1e-4, "elementwise": 1e-3, "conv_ops": 5e-3, "const": 0.05}
    points = []
    for i, (macs, ew, conv) in enumerate(
        [(11776, 729, 71), (1731, 274, 71), (6000, 500, 140), (3000, 300, 35),
         (9000, 600, 200), (4500, 400, 100)]
    ):
        f = {"macs": float(macs), "elementwise": float(ew), "conv_ops": float(conv), "const": 1.0}
        points.append(Measurement(f"m{i}", f, sum(truth[k] * f[k] for k in truth)))

    model = fit(points)
    assert model.is_adequate
    assert model.identifiable == {"macs", "elementwise", "conv_ops", "const"}
    for key, expected in truth.items():
        assert model.coefficients[key] == pytest.approx(expected, rel=1e-6)


def test_uncalibrated_model_predicts_zero_rather_than_guessing():
    assert DEFAULT_MODEL.predict(features_of(_report())) == 0.0


def test_features_extracted_from_cost_report():
    f = features_of(_report())
    assert f == {"macs": 11776.0, "elementwise": 729.0, "conv_ops": 71.0, "const": 1.0}


def test_uncalibrated_model_does_not_gate():
    """An unidentified model must not be used as a hard constraint.

    Gating on it would be no more correct than gating on raw MACs, and much harder
    to see through when a result looks wrong.
    """
    budget = Budget.from_reference(_report(), DEFAULT_MODEL)
    assert budget.model_gates is False
    assert budget.predicted_us_per_sample is not None  # still reported

    # A model with 3x the layers at identical MACs is not rejected, because the
    # calibration cannot yet tell that depth costs anything.
    deep = _report(conv_ops=213)
    assert check_budget(deep, budget, DEFAULT_MODEL) is None


def test_mac_cap_still_binds_regardless():
    budget = Budget.from_reference(_report(), DEFAULT_MODEL)
    breach = check_budget(_report(macs=20000.0), budget, DEFAULT_MODEL)
    assert breach is not None and "MAC budget" in breach


def test_empty_fit_rejected():
    with pytest.raises(ValueError):
        fit([])
