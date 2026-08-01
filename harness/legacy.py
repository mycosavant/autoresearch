"""
Upgrade legacy ``.nam`` configs. READ-ONLY: not to be modified by the research agent.

Current NAM cannot load its own A1-era model files. ``LayerArray.parse_config``
requires a ``head`` object, while A1-era exports carry the older scalar
``head_size`` / ``head_bias`` pair. Every A1 example model in
``NeuralAmpModelerCore/example_models/`` fails to load as a result, which would
otherwise block the A1 baseline the research program requires before any novelty.

This converts in-memory only. Nothing on disk is rewritten, and NAM itself is not
patched -- the harness is a consumer of NAM, not a fork of it.
"""

from __future__ import annotations

import copy as _copy
from typing import Any, Dict

__all__ = ["upgrade_config", "load_nam"]


class UnsupportedLegacyFeature(ValueError):
    """A legacy config uses something the current schema cannot express."""


def _upgrade_layer_array(layer: Dict[str, Any], index: int) -> Dict[str, Any]:
    layer = _copy.deepcopy(layer)

    if "head" not in layer and "head_size" in layer:
        # The legacy layer-array head was always a 1x1 rechannel convolution; the
        # windowed (kernel_size > 1) head only arrived with A2.
        layer["head"] = {
            "out_channels": layer.pop("head_size"),
            "kernel_size": 1,
            "bias": bool(layer.pop("head_bias", False)),
        }
    layer.pop("head_size", None)
    layer.pop("head_bias", None)

    # `gated` selected a PairMultiply activation. Refuse rather than silently drop:
    # dropping it would produce a model that loads, trains, and is quietly the wrong
    # architecture -- the worst possible failure for a baseline.
    if layer.pop("gated", False):
        raise UnsupportedLegacyFeature(
            f"layer array {index} sets gated=True. Gated activations are expressed "
            f"differently in the current schema (a PairMultiply activation), and are "
            f"not auto-converted here because a wrong conversion would give a "
            f"plausible-looking but incorrect baseline."
        )

    return layer


def upgrade_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Return ``config`` (a ``.nam`` ``config`` object) in the current schema."""
    config = _copy.deepcopy(config)
    if "layers" in config:
        config["layers"] = [
            _upgrade_layer_array(layer, i) for i, layer in enumerate(config["layers"])
        ]
    return config


def load_nam(model_dict: Dict[str, Any]):
    """Instantiate a NAM model from a ``.nam`` dict, upgrading legacy configs.

    :param model_dict: A parsed ``.nam`` file, or one ``SlimmableContainer`` submodel.
    """
    from nam.models import init_from_nam

    upgraded = _copy.deepcopy(model_dict)
    if isinstance(upgraded.get("config"), dict):
        upgraded["config"] = upgrade_config(upgraded["config"])
    return init_from_nam(upgraded)
