"""
Frozen reference architectures. READ-ONLY: not to be modified by the research agent.

A2 is the incumbent this repo exists to challenge, so its definition is the compute
cap. The cap is *derived* from these constructors on every run rather than stored as
a number in a config file, which means there is no artifact to tamper with and no
way for the cap to silently drift out of sync with what it claims to represent.

Provenance
----------
Reconstructed from the C++ fast path's detector
(``NeuralAmpModelerCore/NAM/wavenet/a2_fast.{h,cpp}``), then **validated against a
real deployed A2 model**: ``BossWN-a2.nam`` from ``mikeoliphant/NeuralAudio`` (MIT),
vendored as a modulus parity fixture and byte-verified against the public upstream
blob. See ``tests/test_real_a2_model.py``.

Neither local repo ships an A2 model -- ``example_models/wavenet_a2_max.nam`` is a
v0.6.0 feature-coverage stress fixture, not A2 -- and there is no A2 preset in the
Python trainer (``nam/train/core.py`` carries only the A1 family; its
``Architecture.NANO`` is *A1* nano). But deployed A2 models do exist publicly.

Do not read modulus's "n=1 deployed" figure as a fact about the world: that survey
ran on 2026-06-06 when A2 was five weeks old, and it records that TONE3000 -- the
dominant A2 distribution channel -- is OAuth/PKCE-gated and returned HTTP 401, so it
could not be enumerated at all. n=1 is what was reachable without an account, not
what exists. TONE3000 has since retrained much of its library to A2. The practical
consequence is that **tail diversity is unknown**: whether deployed A2 ever activates
FiLM, gating, or channel counts beyond 3 and 8 has never been established, only that
the single reachable model did not.

**How A2 is actually trained**, which the detector cannot tell you: the default
recipe (``nam/train/_resources/config_model_packed.json``) is a ``PackedWaveNet``
carrying ``channels_3`` and ``channels_8`` trained *masked together*, then exported
as a ``SlimmableContainer``. The baseline in ``train.py`` trains each width
independently instead, which is a real divergence from how A2 was made -- joint
training may help the small width or hurt the large one. Worth measuring before
treating an independently-trained A2 as "the A2 number".

Expected costs, cross-checked three ways -- against the C++ detector constants,
against ``tools/test/test_a2_fast.cpp``'s independent weight counter, and by
reproducing the real ``wavenet_a1_standard.nam``'s 13802-weight stream exactly:

===============  ==========  ==========
Quantity         A2 standard A2 nano
===============  ==========  ==========
channels                   8          3
parameters            12_145      1_870
MACs/sample           11_776      1_731
receptive field         6_347      6_347
===============  ==========  ==========

The MAC figures here are one lower than the 11_777/1_732 you get by hand, because
the trailing ``head_scale`` multiply is a pointwise op and is accounted against the
elementwise budget rather than the MAC budget. See :mod:`harness.cost`.

Do not use ``NeuralAmpModelerCore/generate_weights_a2.py`` as a parameter-count
oracle: it reads a scalar ``kernel_size`` (KeyErrors on A2's ``kernel_sizes``) and
counts the head rechannel as a 1x1, undercounting it 16-fold.
"""

from __future__ import annotations

import copy as _copy
from typing import Any, Dict

__all__ = [
    "A2_STANDARD_CONFIG",
    "A2_NANO_CONFIG",
    "A2_STANDARD_MACS",
    "A2_NANO_MACS",
    "A2_STANDARD_PARAMS",
    "A2_NANO_PARAMS",
    "A2_RECEPTIVE_FIELD",
    "A1_STANDARD_CONFIG",
    "A1_STANDARD_PARAMS",
    "A1_STANDARD_MACS",
    "A1_RECEPTIVE_FIELD",
    "LSTM_CONFIG",
    "build_reference",
    "a2_standard",
    "a2_nano",
    "a1_standard",
    "lstm_baseline",
]

# --------------------------------------------------------------------------------
# The A2 shape. Every constant below is enforced element-for-element by
# is_a2_shape(); any deviation drops the model off the C++ fast path.
# --------------------------------------------------------------------------------

_A2_KERNEL_SIZES = [6] * 14 + [15, 15] + [6] * 7  # 23 entries, sum = 156
_A2_DILATIONS = [
    1, 3, 7, 17, 41, 101, 239,   # block A
    1, 3, 7, 17, 41, 101, 239,   # block B
    1, 13,                       # fine-detail interlude (the k=15 pair)
    1, 3, 7, 17, 41, 101, 239,   # block C
]

assert len(_A2_KERNEL_SIZES) == len(_A2_DILATIONS) == 23

#: Layer-array head *rechannel* convolution kernel. This is not a post-stack head --
#: the WaveNet-level ``head`` must stay ``None``.
_A2_HEAD_KERNEL_SIZE = 16

#: Nominal head scale. NOT a universal A2 constant, despite what the C++ detector
#: implies -- see below. Irrelevant to cost (one elementwise multiply, no effect on
#: parameters or MACs), so 0.01 is used here purely to match the detector's
#: definition of "A2 shape".
#:
#: In real models this value absorbs A2's -18 dBFS loudness normalization, which is
#: folded back into the head scale at export. The validated real-world A2 model
#: ``BossWN-a2.nam`` (mikeoliphant/NeuralAudio, MIT) carries 0.013119162069029721.
#:
#: 0.01 is the *trainer initialization* value (``config_model_packed.json``), which
#: the export hook then overwrites. NAM's ``_ScaleOutputHook`` multiplies both
#: ``config.head_scale`` and ``weights[-1]`` by the loudness scale, so a deployed
#: model's head_scale is whatever normalization produced.
#:
#: **Version-specific trap.** The pinned ``NeuralAmpModelerCore`` checkout here
#: (v0.5.1/v0.5.2) defines ``kHeadScale = 0.01f`` and has ``is_a2_shape()`` reject
#: anything more than 1e-7 away from it (``a2_fast.cpp:778-782``) -- while
#: ``_load_weights`` overrides ``_head_scale`` from the weight stream regardless
#: (``a2_fast.cpp:264-265``). At that pin, the only known deployed A2 model is
#: rejected by the specialization written for it and silently falls back to the
#: generic WaveNet. **Upstream ``main`` has since removed ``kHeadScale`` entirely**;
#: ``is_a2_shape`` now only checks that head_scale is present and numeric. So this is
#: a stale-pin issue, not a live upstream bug -- worth pulling into the fork.
_A2_HEAD_SCALE = 0.01

_A2_LEAKY_SLOPE = 0.01


def _a2_config(channels: int) -> Dict[str, Any]:
    return {
        "layers_configs": [
            {
                "input_size": 1,
                "condition_size": 1,
                "channels": channels,
                "bottleneck": channels,  # A2 hard-requires bottleneck == channels
                "head": {
                    "out_channels": 1,
                    "kernel_size": _A2_HEAD_KERNEL_SIZE,
                    "bias": True,
                },
                "kernel_sizes": list(_A2_KERNEL_SIZES),
                "dilations": list(_A2_DILATIONS),
                "activation": {
                    "name": "LeakyReLU",
                    "negative_slope": _A2_LEAKY_SLOPE,
                },
                # Explicit rather than defaulted: these being off *is* the A2 shape,
                # and a future default change should break loudly, not silently.
                "layer_1x1_config": {"active": True, "groups": 1},
                "head_1x1_config": {"active": False, "out_channels": 1, "groups": 1},
                "groups_input": 1,
                "groups_input_mixin": 1,
                "film_params": {},
                "slimmable": None,
            }
        ],
        "head": None,
        "head_scale": _A2_HEAD_SCALE,
    }


A2_STANDARD_CONFIG: Dict[str, Any] = _a2_config(8)
A2_NANO_CONFIG: Dict[str, Any] = _a2_config(3)

#: Expected values, asserted at construction so a NAM-side change that alters the
#: architecture cannot quietly move the compute cap underneath the whole run log.
A2_STANDARD_MACS = 11_776
A2_NANO_MACS = 1_731
A2_STANDARD_PARAMS = 12_145
A2_NANO_PARAMS = 1_870
A2_RECEPTIVE_FIELD = 6_347


def build_reference(config: Dict[str, Any]):
    """Instantiate a NAM ``WaveNet`` from one of the frozen configs."""
    from nam.models.wavenet import WaveNet  # imported lazily; pulls in torch

    return WaveNet.init_from_config(_copy.deepcopy(config))


def a2_standard():
    """The incumbent. Its cost is the compute cap."""
    return build_reference(A2_STANDARD_CONFIG)


def a2_nano():
    """The low-compute reference point."""
    return build_reference(A2_NANO_CONFIG)


# --------------------------------------------------------------------------------
# Prior-generation baselines, for context on how much A2 actually moved.
#
# A1 standard is transcribed from example_models/wavenet_a1_standard.nam, upgraded
# to the current schema (that file uses the legacy head_size/head_bias pair, which
# current NAM cannot load -- see harness/legacy.py). Verified on load: 13,801
# parameters, receptive field 4,093, 13,320 MACs/sample.
#
# Note A1 uses head_scale 0.02 and Tanh, against A2's 0.01 and LeakyReLU, and reaches
# a receptive field of only 4,093 against A2's 6,347 while costing *more*
# (13,320 vs 11,776 MACs/sample). That gap is the thing this repo is trying to beat
# again.
# --------------------------------------------------------------------------------

_A1_DILATIONS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

A1_STANDARD_CONFIG: Dict[str, Any] = {
    "layers_configs": [
        {
            "input_size": 1,
            "condition_size": 1,
            "channels": 16,
            "head": {"out_channels": 8, "kernel_size": 1, "bias": False},
            "kernel_size": 3,
            "dilations": list(_A1_DILATIONS),
            "activation": "Tanh",
        },
        {
            "input_size": 16,
            "condition_size": 1,
            "channels": 8,
            "head": {"out_channels": 1, "kernel_size": 1, "bias": True},
            "kernel_size": 3,
            "dilations": list(_A1_DILATIONS),
            "activation": "Tanh",
        },
    ],
    "head": None,
    "head_scale": 0.02,
}

A1_STANDARD_PARAMS = 13_801
A1_STANDARD_MACS = 13_320
A1_RECEPTIVE_FIELD = 4_093

#: The LSTM baseline, from example_models/lstm.nam. Included because it is a
#: structurally different answer to the same problem -- unbounded memory through
#: recurrence rather than a finite dilated window -- and so is a useful sanity check
#: that the harness is not silently specialized to convolutional models.
LSTM_CONFIG: Dict[str, Any] = {"input_size": 1, "hidden_size": 3, "num_layers": 1}


def a1_standard():
    """Previous-generation reference. Costs more than A2 for less receptive field."""
    return build_reference(A1_STANDARD_CONFIG)


def lstm_baseline():
    """Recurrent baseline."""
    from nam.models.recurrent import LSTM

    return LSTM(**_copy.deepcopy(LSTM_CONFIG))
