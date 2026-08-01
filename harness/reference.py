"""
Frozen reference architectures. READ-ONLY: not to be modified by the research agent.

A2 is the incumbent this repo exists to challenge, so its definition is the compute
cap. The cap is *derived* from these constructors on every run rather than stored as
a number in a config file, which means there is no artifact to tamper with and no
way for the cap to silently drift out of sync with what it claims to represent.

Provenance
----------
Reconstructed from the C++ fast path's detector, which is the only authoritative A2
definition that exists in either repo:
``NeuralAmpModelerCore/NAM/wavenet/a2_fast.{h,cpp}``. Note that there is **no
shipped A2 .nam file** -- ``example_models/wavenet_a2_max.nam`` is a feature-coverage
stress model for the v0.6.0 schema, not an A2 model, and ``is_a2_shape()`` rejects
it. There is also no A2 preset in the Python trainer (``nam/train/core.py`` carries
only the A1 family; its ``Architecture.NANO`` is *A1* nano and is unrelated).

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
    "build_reference",
    "a2_standard",
    "a2_nano",
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
#: That exposes a genuine inconsistency in the C++ fast path: ``is_a2_shape()``
#: gates on the JSON ``head_scale`` being within 1e-7 of 0.01
#: (``a2_fast.cpp:778-782``), but ``_load_weights`` then overrides ``_head_scale``
#: from the trailing value in the weight stream anyway (``a2_fast.cpp:264-265``).
#: So the field is load-bearing for *detection* while being ignored for *computation*
#: -- and a real A2 model whose loudness normalization moved head_scale off 0.01 is
#: silently rejected by the specialization built for it, falling back to the generic
#: WaveNet with no warning. Verified against BossWN-a2; whether it holds across a
#: corpus of A2 models has not been checked.
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
