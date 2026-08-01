"""
Generate WaveNet variants for cost-model calibration.

The shipped calibration is degenerate: A2 Full and A2 Lite are structurally
identical (both 23 layers, 71 convolutions) and differ only in width, so the
per-convolution coefficient cannot be separated from the intercept. Fixing that
needs architectures whose *depth* varies independently of their *width*.

This emits exactly that spread, as ``.nam`` files that modulus can load and
benchmark. Weights are untrained and left at initialization -- these are timing
vehicles, not models. That is sound here because none of the kernels involved
branch on weight values, so runtime is a function of shape alone. (It would not be
sound for anything with data-dependent control flow, e.g. early exit or sparsity.)

Usage::

    python tools/make_calibration_models.py --out /tmp/calib

Then benchmark each with modulus and add the points to
``harness.cost_model.MODULUS_MEASUREMENTS``. See docs/CALIBRATION.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.cost import count_cost  # noqa: E402
from harness.cost_model import features_of  # noqa: E402
from harness.reference import (  # noqa: E402
    A2_STANDARD_CONFIG,
    build_reference,
)

#: A2's dilation cycle. Repeated to reach an arbitrary depth so that variants stay
#: in the same architectural family as the reference -- we are varying depth and
#: width, not inventing a different kind of network.
_CYCLE = [1, 3, 7, 17, 41, 101, 239]


def variant(name: str, n_layers: int, channels: int, kernel_size: int = 6) -> Dict:
    """A single-layer-array WaveNet of the A2 family at a chosen depth and width."""
    config = copy.deepcopy(A2_STANDARD_CONFIG)
    layer = config["layers_configs"][0]
    layer["channels"] = channels
    layer["bottleneck"] = channels
    layer["kernel_sizes"] = [kernel_size] * n_layers
    layer["dilations"] = [_CYCLE[i % len(_CYCLE)] for i in range(n_layers)]
    return config


#: Chosen so that conv_ops and macs vary as independently as possible. Depth drives
#: conv_ops (3 per layer plus rechannel and head); width drives macs quadratically.
#: The deep-narrow and shallow-wide corners are what break the collinearity.
VARIANTS = [
    ("a2_full", 23, 8),          # the reference point
    ("a2_lite", 23, 3),          # reference, narrow
    ("deep_narrow", 46, 3),      # 2x depth, low arithmetic
    ("very_deep_narrow", 46, 2),
    ("shallow_wide", 12, 16),    # high arithmetic, few ops
    ("shallow_narrow", 8, 8),
    ("mid", 23, 5),
    ("deep_wide", 34, 6),
]


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("/tmp/calib"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    print(f"{'name':20s} {'layers':>6s} {'ch':>3s} {'macs':>8s} {'conv':>5s} {'ew':>7s} {'params':>8s}")
    print("-" * 66)

    for name, n_layers, channels in VARIANTS:
        config = variant(name, n_layers, channels)
        model = build_reference(config)
        report = count_cost(model, input_samples=32768)
        features = features_of(report)

        model.export(args.out, basename=name)

        manifest.append(
            {
                "name": name,
                "layers": n_layers,
                "channels": channels,
                "file": f"{name}.nam",
                "features": features,
                "params": report.params,
                "receptive_field": int(model.receptive_field),
            }
        )
        print(
            f"{name:20s} {n_layers:6d} {channels:3d} {features['macs']:8.0f} "
            f"{features['conv_ops']:5.0f} {features['elementwise']:7.1f} {report.params:8d}"
        )

    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    convs = sorted({m["features"]["conv_ops"] for m in manifest})
    macs = sorted({m["features"]["macs"] for m in manifest})
    print(f"\ndistinct conv_ops: {convs}")
    print(f"distinct macs:     {len(macs)} values, {min(macs):.0f}..{max(macs):.0f}")
    print(f"\nwrote {len(manifest)} models + manifest.json to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
