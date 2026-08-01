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

This matters concretely. The ``modulus-i9-14900f`` profile has two points of
*identical structure* -- both 23 layers and 71 convolutions -- differing only in
width, so it cannot separate a per-convolution overhead term from a constant. It
reports itself unidentifiable rather than returning coefficients that merely fit.
The ``devcontainer-4cpu`` profile shows the fix: eight architectures spanning
26-140 convolutions and 1.4k-22k MACs, which identifies all four coefficients.

Calibration is per-machine
--------------------------
Coefficients do not transfer between machines -- they encode a specific CPU's
arithmetic throughput and per-call overhead. So calibrations are stored as named
*profiles* under ``harness/calibration/``, and the active one is chosen explicitly by
:data:`~harness.constants.CALIBRATION_PROFILE`.

If no profile is selected, the model is uncalibrated: predictions are unavailable and
nothing gates. That is deliberate. Silently applying another machine's coefficients
would produce confident, wrong numbers -- worse than no model at all.

Run ``tools/calibrate.py`` on a new machine to produce its profile.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass, field as _field
from pathlib import Path as _Path
from typing import Dict, List, Mapping, Sequence, Set

import numpy as _np

from .cost import CostReport

__all__ = [
    "FEATURES",
    "Measurement",
    "CostModel",
    "features_of",
    "fit",
    "load_profile",
    "available_profiles",
    "active_model",
    "DEFAULT_MODEL",
    "CALIBRATION_DIR",
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
# Profiles
# --------------------------------------------------------------------------------

CALIBRATION_DIR = _Path(__file__).resolve().parent / "calibration"


def available_profiles() -> List[str]:
    """Names of calibration profiles present on disk."""
    if not CALIBRATION_DIR.is_dir():
        return []
    return sorted(p.stem for p in CALIBRATION_DIR.glob("*.json"))


def load_profile(name: str) -> CostModel:
    """Load and fit a named calibration profile."""
    path = CALIBRATION_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"No calibration profile {name!r} in {CALIBRATION_DIR}. "
            f"Available: {available_profiles() or 'none'}. "
            f"Run tools/calibrate.py to create one for this machine."
        )
    data = _json.loads(path.read_text())
    measurements = [
        Measurement(
            name=m["name"],
            features=m["features"],
            microseconds_per_sample=float(m["microseconds_per_sample"]),
            provenance=data.get("tool", ""),
        )
        for m in data["measurements"]
    ]
    model = fit(measurements, provenance=f"{data.get('profile', name)}: {data.get('machine', '')}")
    if data.get("caveat"):
        model.notes.append(data["caveat"])
    return model


def _uncalibrated() -> CostModel:
    """A model that knows it has no calibration, and therefore predicts nothing."""
    return CostModel(
        coefficients={f: 0.0 for f in FEATURES},
        identifiable=set(),
        n_measurements=0,
        condition_number=float("inf"),
        residual_rel_error=float("inf"),
        provenance="uncalibrated",
        notes=[
            "No calibration profile selected. Set CALIBRATION_PROFILE in "
            "harness/constants.py to one of "
            f"{available_profiles() or ['(none available)']}, or run tools/calibrate.py "
            "to measure this machine. Predictions are unavailable and the "
            "runtime cap does not gate; only the MAC and elementwise caps bind."
        ],
    )


def active_model() -> CostModel:
    """The calibration for this machine, or an uncalibrated model if none is set."""
    from .constants import CALIBRATION_PROFILE

    if not CALIBRATION_PROFILE:
        return _uncalibrated()
    try:
        return load_profile(CALIBRATION_PROFILE)
    except FileNotFoundError as e:  # pragma: no cover - configuration error
        model = _uncalibrated()
        model.notes.append(str(e))
        return model


DEFAULT_MODEL = active_model()
