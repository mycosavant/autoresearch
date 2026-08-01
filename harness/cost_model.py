"""
Calibrated runtime cost model. READ-ONLY: not to be modified by the research agent.

Why this exists
---------------
MACs alone mis-rank architectures at NAM's channel widths. Measured on real A2
models, going from 3 to 8 channels is **6.8x more MACs but only 2.17x more time**
(1.61x in C++) -- at Lite width the majority of the runtime is not arithmetic at all.
The dilated convolution is ~85% of the MACs but its inner reduction is only 3-8 wide,
so the work is dominated by fixed per-operation overhead rather than throughput.

A search scored purely on MACs therefore has a known bias: it systematically
undervalues width and overvalues depth, because per-layer overhead is invisible to
it. This module replaces the raw MAC count with a linear model over *structural*
features, fitted to real measurements.

Honesty about the fit
---------------------
The model refuses to pretend it is identified when it is not. Fitting is done by
least squares with an explicit rank and conditioning check: any coefficient that the
available measurements cannot separate is reported as **unidentifiable** rather than
handed back as a number that happens to minimise residuals.

This matters immediately. The two shipped calibration points (A2 Full and A2 Lite)
have *identical structure* -- both are 23 layers and 71 convolutions -- and differ
only in width. They therefore cannot distinguish a per-convolution overhead term from
a constant per-call term. Getting a real per-layer coefficient needs measurements at
differing depths; see :func:`fit` and ``docs/CALIBRATION.md``.

Until then :data:`DEFAULT_MODEL` reports ``is_adequate == False``, and the runner
treats its prediction as advisory rather than as a gate.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass, field as _field
from typing import Dict, List, Mapping, Sequence, Set

import numpy as _np

from .cost import CostReport

__all__ = [
    "FEATURES",
    "Measurement",
    "CostModel",
    "features_of",
    "fit",
    "DEFAULT_MODEL",
    "MODULUS_MEASUREMENTS",
]

#: Structural features the runtime is modelled against. ``const`` is the intercept:
#: fixed per-call cost that scales with neither arithmetic nor depth.
FEATURES = ("macs", "elementwise", "conv_ops", "const")

#: Condition-number ceiling above which the design matrix is treated as degenerate.
_CONDITION_LIMIT = 1e8

#: Minimum measurements before a fit is considered adequate to gate on. Two points
#: can only ever fit two coefficients exactly, which is interpolation, not evidence.
_MIN_MEASUREMENTS = 5


@_dataclass(frozen=True)
class Measurement:
    """One benchmarked architecture: structural features and its measured cost."""

    name: str
    features: Mapping[str, float]
    microseconds_per_sample: float
    provenance: str = ""


@_dataclass
class CostModel:
    """Linear cost model over :data:`FEATURES`, in microseconds per output sample."""

    coefficients: Dict[str, float]
    identifiable: Set[str]
    n_measurements: int
    condition_number: float
    residual_rel_error: float
    provenance: str = ""
    notes: List[str] = _field(default_factory=list)

    @property
    def is_adequate(self) -> bool:
        """Whether this fit should be gated on, as opposed to merely reported."""
        return (
            self.n_measurements >= _MIN_MEASUREMENTS
            and self.identifiable == set(FEATURES)
            and self.condition_number < _CONDITION_LIMIT
        )

    def predict(self, features: Mapping[str, float]) -> float:
        """Predicted microseconds per output sample."""
        return float(
            sum(self.coefficients.get(f, 0.0) * features.get(f, 0.0) for f in FEATURES)
        )

    def explain(self) -> str:
        lines = [
            f"cost model ({self.n_measurements} measurements, "
            f"cond={self.condition_number:.3g}, "
            f"rel. residual={self.residual_rel_error:.3%})",
            f"  adequate to gate on: {self.is_adequate}",
        ]
        for f in FEATURES:
            mark = "" if f in self.identifiable else "   [UNIDENTIFIABLE]"
            lines.append(f"  {f:12s} {self.coefficients.get(f, 0.0):+.6g}{mark}")
        lines.extend(f"  note: {n}" for n in self.notes)
        return "\n".join(lines)


def features_of(report: CostReport) -> Dict[str, float]:
    """Extract model features from a measured :class:`CostReport`."""
    conv_ops = sum(v for k, v in report.op_counts.items() if "conv" in k)
    return {
        "macs": float(report.macs_per_sample),
        "elementwise": float(report.elementwise_per_sample),
        "conv_ops": float(conv_ops),
        "const": 1.0,
    }


def fit(measurements: Sequence[Measurement], provenance: str = "") -> CostModel:
    """Least-squares fit with an explicit identifiability check.

    A feature is reported identifiable only if the measurements actually vary it
    independently of the others. Constant or collinear columns are excluded from the
    identifiable set, so a degenerate calibration announces itself instead of
    returning coefficients that merely happen to fit.
    """
    if not measurements:
        raise ValueError("Cannot fit a cost model with no measurements.")

    design = _np.array(
        [[float(m.features.get(f, 0.0)) for f in FEATURES] for m in measurements],
        dtype=float,
    )
    target = _np.array([m.microseconds_per_sample for m in measurements], dtype=float)

    solution, *_ = _np.linalg.lstsq(design, target, rcond=None)
    coefficients = {f: float(v) for f, v in zip(FEATURES, solution)}

    notes: List[str] = []
    identifiable: Set[str] = set()

    # A column that never varies (other than the intercept) carries no information
    # about its own coefficient -- its effect is absorbed into `const`.
    for i, f in enumerate(FEATURES):
        column = design[:, i]
        if f != "const" and float(_np.ptp(column)) == 0.0:
            notes.append(
                f"{f!r} is constant across all measurements, so its coefficient is "
                f"indistinguishable from the intercept; vary it to identify."
            )
            continue
        identifiable.add(f)

    rank = int(_np.linalg.matrix_rank(design))
    if rank < design.shape[1]:
        notes.append(
            f"design matrix is rank-deficient ({rank} < {design.shape[1]}): some "
            f"features are collinear and cannot be separated."
        )
        identifiable &= set()

    singular = _np.linalg.svd(design, compute_uv=False)
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else _np.inf

    predicted = design @ solution
    denom = _np.maximum(_np.abs(target), 1e-12)
    residual = float(_np.max(_np.abs(predicted - target) / denom))

    if len(measurements) < _MIN_MEASUREMENTS:
        notes.append(
            f"only {len(measurements)} measurements; at least {_MIN_MEASUREMENTS} "
            f"with differing depths are needed before this should gate anything."
        )

    return CostModel(
        coefficients=coefficients,
        identifiable=identifiable,
        n_measurements=len(measurements),
        condition_number=condition,
        residual_rel_error=residual,
        provenance=provenance,
        notes=notes,
    )


# --------------------------------------------------------------------------------
# Shipped calibration.
#
# Source: modulus `.claude/research/2026-06-06-nam-a2-perf-baseline.md` and
# `crates/inference-nam/CLAUDE.md`, measured on an i9-14900F, release build, block
# size 64, `process()`-only mean, after the frame-axis vectorization work.
# Converted from microseconds-per-64-sample-block to microseconds-per-sample.
#
# Both points are 23 layers / 71 convolutions, so `conv_ops` does not vary and its
# coefficient is not identifiable from them. This is deliberately left visible.
# --------------------------------------------------------------------------------

_BLOCK = 64.0

MODULUS_MEASUREMENTS: List[Measurement] = [
    Measurement(
        name="a2_full_8ch",
        features={"macs": 11776.0, "elementwise": 729.0, "conv_ops": 71.0, "const": 1.0},
        microseconds_per_sample=202.0 / _BLOCK,
        provenance="modulus Rust inference-nam, i9-14900F, block 64",
    ),
    Measurement(
        name="a2_lite_3ch",
        features={"macs": 1731.0, "elementwise": 274.0, "conv_ops": 71.0, "const": 1.0},
        microseconds_per_sample=55.0 / _BLOCK,
        provenance="modulus Rust inference-nam, i9-14900F, block 64",
    ),
]

DEFAULT_MODEL = fit(
    MODULUS_MEASUREMENTS,
    provenance="modulus A2 perf baseline (i9-14900F, block 64, Rust)",
)
