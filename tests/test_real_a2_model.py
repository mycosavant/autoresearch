"""Validate the A2 reconstruction against a real, in-the-wild A2 model file.

This is the strongest available check on `harness/reference.py`. The reference was
reconstructed from the C++ fast-path detector rather than transcribed from a spec, so
agreement with an actual shipped A2 model -- field for field, and to the exact weight
count -- is what rules out a plausible-but-wrong reconstruction.

Fixture: `BossWN-a2.nam` from mikeoliphant/NeuralAudio (MIT), vendored into the
modulus repo's parity fixtures. Skipped when that repo is not present.
"""

import json
from pathlib import Path

import pytest

from harness.reference import (
    A2_NANO_CONFIG,
    A2_NANO_PARAMS,
    A2_STANDARD_CONFIG,
    A2_STANDARD_PARAMS,
)

FIXTURE = Path(
    "/workspace/modulus/crates/test-harness/tests/fixtures/nam-a2/BossWN-a2.nam"
)

pytestmark = pytest.mark.skipif(
    not FIXTURE.is_file(), reason="modulus A2 fixture not available in this checkout"
)


def _submodels():
    data = json.loads(FIXTURE.read_text())
    assert data["architecture"] == "SlimmableContainer"
    return [s["model"] for s in data["config"]["submodels"]]


@pytest.fixture(scope="module")
def models():
    return {m["config"]["layers"][0]["channels"]: m for m in _submodels()}


@pytest.mark.parametrize(
    "channels,reference,expected_params",
    [(8, A2_STANDARD_CONFIG, A2_STANDARD_PARAMS), (3, A2_NANO_CONFIG, A2_NANO_PARAMS)],
    ids=["standard", "nano"],
)
def test_real_model_matches_reconstruction(models, channels, reference, expected_params):
    model = models[channels]
    config = model["config"]
    layer = config["layers"][0]
    ref_layer = reference["layers_configs"][0]

    assert len(config["layers"]) == 1, "A2 is a single layer array"
    assert config["head"] is None, "A2 has no post-stack head"
    assert layer["kernel_sizes"] == ref_layer["kernel_sizes"]
    assert layer["dilations"] == ref_layer["dilations"]
    assert layer["head"] == ref_layer["head"]
    assert layer["bottleneck"] == ref_layer["bottleneck"] == channels
    assert layer["slimmable"] is None

    activations = layer["activation"]
    assert len(activations) == 23
    assert all(
        a["type"] == "LeakyReLU" and a["negative_slope"] == 0.01 for a in activations
    )

    # The decisive check: the weight stream is parameters plus one trailing
    # head_scale float. Getting this exactly right means every per-layer shape in the
    # reconstruction is correct, not merely the ones spelled out in the config.
    assert len(model["weights"]) == expected_params + 1


def test_real_model_head_scale_is_not_the_detector_constant(models):
    """head_scale is per-model, not an architectural constant.

    It absorbs A2's -18 dBFS loudness normalization. The C++ `is_a2_shape()` gates on
    it being ~0.01 while `_load_weights` overrides it from the weight stream anyway,
    so a real A2 model like this one is rejected by the specialization built for it
    and silently falls back to the generic WaveNet.
    """
    for model in models.values():
        head_scale = model["config"]["head_scale"]
        assert abs(head_scale - 0.01) > 1e-7, (
            "This fixture is expected to carry a loudness-compensated head_scale. If "
            "it now equals the detector constant, re-check the fast-path analysis in "
            "harness/reference.py."
        )
