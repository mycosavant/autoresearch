"""A2 reference tests.

The compute cap *is* A2's cost, so if these drift the whole run log becomes
incomparable. Expected values were derived independently from the C++ fast-path
detector (NeuralAmpModelerCore/NAM/wavenet/a2_fast.{h,cpp}) and cross-checked
against that repo's own weight counter.
"""

import pytest

from harness.cost import count_cost
from harness.reference import (
    A2_NANO_MACS,
    A2_NANO_PARAMS,
    A2_RECEPTIVE_FIELD,
    A2_STANDARD_MACS,
    A2_STANDARD_PARAMS,
    a2_nano,
    a2_standard,
)


@pytest.mark.parametrize(
    "build,expected_params,expected_macs",
    [
        (a2_standard, A2_STANDARD_PARAMS, A2_STANDARD_MACS),
        (a2_nano, A2_NANO_PARAMS, A2_NANO_MACS),
    ],
    ids=["standard", "nano"],
)
def test_a2_matches_reference_costs(build, expected_params, expected_macs):
    model = build()
    assert sum(p.numel() for p in model.parameters()) == expected_params
    assert model.receptive_field == A2_RECEPTIVE_FIELD

    report = count_cost(model, input_samples=32768)
    assert report.macs_per_sample == pytest.approx(expected_macs, abs=0.5)
    assert report.is_linear


def test_nano_differs_from_standard_only_in_width():
    from harness.reference import A2_NANO_CONFIG, A2_STANDARD_CONFIG

    standard = A2_STANDARD_CONFIG["layers_configs"][0]
    nano = A2_NANO_CONFIG["layers_configs"][0]
    differing = {k for k in standard if standard[k] != nano.get(k)}
    assert differing == {"channels", "bottleneck"}


def test_a2_head_scale_is_not_the_a1_value():
    """A2 uses 0.01; A1 uses 0.02. Getting this wrong silently drops A2 off the
    C++ fast path with no warning."""
    from harness.reference import A2_STANDARD_CONFIG

    assert A2_STANDARD_CONFIG["head_scale"] == 0.01
