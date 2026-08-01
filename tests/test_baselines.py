"""Baseline architecture tests.

`program.md` requires A2 standard, A2 nano, A1 standard and LSTM to be reproduced and
logged before the search invents anything. All four must therefore build and cost
correctly, or the run log has no reference points.
"""

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from harness.cost import count_cost
from harness.cost_model import features_of
from harness.legacy import UnsupportedLegacyFeature, load_nam, upgrade_config
from harness.reference import (
    A1_RECEPTIVE_FIELD,
    A1_STANDARD_MACS,
    A1_STANDARD_PARAMS,
    a1_standard,
    lstm_baseline,
)

A1_NAM = Path("/home/user/NeuralAmpModelerCore/example_models/wavenet_a1_standard.nam")


def test_a1_standard_matches_reference_costs():
    model = a1_standard()
    report = count_cost(model, input_samples=32768)
    assert report.params == A1_STANDARD_PARAMS
    assert model.receptive_field == A1_RECEPTIVE_FIELD
    assert report.macs_per_sample == pytest.approx(A1_STANDARD_MACS, abs=0.5)


def test_a1_costs_more_than_a2_for_less_receptive_field():
    """The claim A2 makes, and the one the search has to beat again."""
    from harness.reference import (
        A2_RECEPTIVE_FIELD,
        A2_STANDARD_MACS,
    )

    assert A1_STANDARD_MACS > A2_STANDARD_MACS
    assert A1_RECEPTIVE_FIELD < A2_RECEPTIVE_FIELD


def test_lstm_baseline_costs_are_counted():
    """Recurrence must not read as free; it used to raise on mkldnn_rnn_layer."""
    model = lstm_baseline()
    report = count_cost(model, input_samples=8192)
    assert report.macs_per_sample > 0
    assert report.is_linear


@pytest.mark.parametrize(
    "hidden,layers",
    [(16, 1), (16, 2), (24, 1)],
    ids=["h16l1", "h16l2", "h24l1"],
)
def test_lstm_macs_match_hand_calculation(hidden, layers):
    """MACs per step = 4H*(input + H) per layer, plus the output projection."""

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, layers, batch_first=True)
            self.head = nn.Linear(hidden, 1)
            self.receptive_field = 1

        def forward(self, x):
            y, _ = self.lstm(x.unsqueeze(-1))
            return self.head(y).squeeze(-1)

    expected = 4 * hidden * 1 + 4 * hidden * hidden
    expected += (layers - 1) * (4 * hidden * hidden + 4 * hidden * hidden)
    expected += hidden

    report = count_cost(Net(), input_samples=4096)
    assert report.macs_per_sample == pytest.approx(expected, abs=0.5)


# --------------------------------------------------------------------------------
# Legacy schema
# --------------------------------------------------------------------------------


def test_upgrade_converts_head_size_to_head_object():
    legacy = {"layers": [{"head_size": 8, "head_bias": False, "channels": 16}]}
    upgraded = upgrade_config(legacy)
    layer = upgraded["layers"][0]
    assert layer["head"] == {"out_channels": 8, "kernel_size": 1, "bias": False}
    assert "head_size" not in layer and "head_bias" not in layer


def test_upgrade_refuses_gated_rather_than_dropping_it():
    """Silently dropping `gated` would yield a plausible but wrong baseline."""
    with pytest.raises(UnsupportedLegacyFeature, match="gated"):
        upgrade_config({"layers": [{"head_size": 8, "head_bias": True, "gated": True}]})


def test_upgrade_leaves_current_schema_untouched():
    from harness.reference import A2_STANDARD_CONFIG

    current = {"layers": [{"head": {"out_channels": 1, "kernel_size": 16, "bias": True}}]}
    assert upgrade_config(current) == current
    # And the A2 config, which is already current, round-trips unchanged.
    assert upgrade_config({"layers": []})["layers"] == []
    assert A2_STANDARD_CONFIG["head_scale"] == 0.01


@pytest.mark.skipif(not A1_NAM.is_file(), reason="NeuralAmpModelerCore not available")
def test_legacy_a1_nam_loads_and_matches_the_transcribed_config():
    """Current NAM cannot load this file unaided -- LayerArray requires a head object.

    Loading it and getting the same numbers as the transcribed config is what
    confirms the transcription in reference.py is faithful.
    """
    loaded = load_nam(json.loads(A1_NAM.read_text()))
    report = count_cost(loaded, input_samples=32768)

    assert report.params == A1_STANDARD_PARAMS
    assert loaded.receptive_field == A1_RECEPTIVE_FIELD
    assert report.macs_per_sample == pytest.approx(A1_STANDARD_MACS, abs=0.5)
    assert features_of(report) == features_of(
        count_cost(a1_standard(), input_samples=32768)
    )
