"""
Export a ``wh_ssm`` model in deployment form, plus golden vectors.

This is the bridge to the Rust side. A ``ModelBackend`` implementation in modulus
has to reproduce this model's output bit-closely enough to pass a parity gate, and
that needs two things: the weights in a form a real-time backend can consume, and a
reference input/output pair to check against.

Export-time folding
-------------------
The file carries the *streaming* parameterization, not the training one:

* ``a00 a01 a10 a11`` instead of ``r_logit`` and ``theta`` -- the 2x2 state matrix
  ``r R(theta)``, with the ``WH_R_MAX * sigmoid`` bound already applied.
* ``b`` with the ``sqrt(1 - r^2)`` drive gain already multiplied in.

Both are functions of the parameters alone, so folding them here moves work out of
the per-sample path and changes nothing numerically. This is the same
re-parameterization ``harness.streaming`` explicitly permits, and for the same
reason: a real export step is allowed to precompute coefficients, and charging a
deployed model for algebra it does once is simply wrong.

Format
------
JSON. The whole model is ~10k parameters, so a binary format would buy nothing but
endianness questions, and a golden fixture that a human can open and read is worth
more than the bytes. Layout is documented in ``SCHEMA`` below.

Usage::

    uv run tools/export_wh_ssm.py --out build/wh_ssm.json --golden 2048
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train as T  # noqa: E402

#: Bump when the layout changes in a way a reader must notice.
SCHEMA = 1

#: Deterministic probe signal for the golden vectors. A fixed seed rather than a
#: sine or an impulse: an impulse never leaves the small-signal region, so it would
#: pass parity on a backend that got the static nonlinearities wrong.
GOLDEN_SEED = 1234
GOLDEN_AMPLITUDE = 0.5


def _flat(t: torch.Tensor) -> List[float]:
    return [float(v) for v in t.detach().reshape(-1)]


def export_bank(bank: T.ResonantBank) -> Dict[str, Any]:
    """One resonant bank, in streaming form.

    Shapes, all row-major:

    ``b``       (n_blocks, 2, in_ch)   drive, gain folded in
    ``a**``     (n_blocks,)            state matrix entries
    ``c_w``     (out_ch, 2 * n_blocks) readout; state index is ``2 * block + part``
    ``c_b``     (out_ch,)
    """
    with torch.no_grad():
        r = bank._radius()
        cos, sin = torch.cos(bank.theta), torch.sin(bank.theta)
        gain = torch.sqrt(1.0 - r * r)[:, None, None]
        return {
            "n_blocks": bank.n_blocks,
            "in_ch": bank.in_ch,
            "out_ch": bank.out_ch,
            "b": _flat(bank.b_proj * gain),
            "a00": _flat(r * cos),
            "a01": _flat(-r * sin),
            "a10": _flat(r * sin),
            "a11": _flat(r * cos),
            "c_w": _flat(bank.c_proj.weight),
            "c_b": _flat(bank.c_proj.bias),
        }


def export_nonlinearity(nl: T.StaticNonlinearity) -> Dict[str, Any]:
    """``fc2(tanh(fc1(x)))``, pointwise in time."""
    with torch.no_grad():
        return {
            "channels": nl.fc1.in_features,
            "hidden": nl.fc1.out_features,
            "w1": _flat(nl.fc1.weight),
            "b1": _flat(nl.fc1.bias),
            "w2": _flat(nl.fc2.weight),
            "b2": _flat(nl.fc2.bias),
        }


def export_model(model: T.WienerHammersteinSSM, golden_samples: int) -> Dict[str, Any]:
    g = torch.Generator().manual_seed(GOLDEN_SEED)
    x = (torch.rand(1, golden_samples, generator=g) * 2.0 - 1.0) * GOLDEN_AMPLITUDE
    with torch.no_grad():
        # The recurrent form specifically: it is what the backend implements, and
        # comparing against the scan would fold this harness's own scan error into
        # the parity budget.
        y = model._run(x, streaming=True)

    return {
        "schema": SCHEMA,
        "architecture": "wh_ssm",
        "sample_rate": 48_000,
        "receptive_field": int(model.receptive_field),
        "stages": [
            {"kind": "bank", **export_bank(model.stage1)},
            {"kind": "nonlinearity", **export_nonlinearity(model.nl1)},
            {"kind": "bank", **export_bank(model.stage2)},
            {"kind": "nonlinearity", **export_nonlinearity(model.nl2)},
            {"kind": "bank", **export_bank(model.stage3)},
        ],
        "golden": {
            "seed": GOLDEN_SEED,
            "amplitude": GOLDEN_AMPLITUDE,
            "input": _flat(x),
            "output": _flat(y),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="model init seed")
    ap.add_argument(
        "--golden",
        type=int,
        default=2048,
        help="golden vector length in samples; 0 to omit",
    )
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    model = T.WienerHammersteinSSM().eval()

    payload = export_model(model, args.golden)
    if args.golden == 0:
        payload.pop("golden")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"wrote {args.out} ({args.out.stat().st_size / 1024:.0f} KiB)")
    print(f"  {n_params} parameters, {len(payload['stages'])} stages")
    if args.golden:
        print(f"  golden: {args.golden} samples")


if __name__ == "__main__":
    main()
