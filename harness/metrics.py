"""
Accuracy metrics. READ-ONLY: not to be modified by the research agent.

Loss definitions are imported from NAM itself rather than reimplemented, so that
numbers out of this harness mean the same thing as numbers out of NAM. If NAM
changes its definition of ESR, this harness changes with it.

On the choice of headline metric
--------------------------------
ESR is NAM's ground truth and is what published A2 numbers are quoted in, so it is
the headline here. It is also a known-imperfect proxy for perceived tone: two models
with equal ESR can sound audibly different, and an optimizer pushed hard enough will
find that gap. So every run also logs multi-resolution STFT error, and
:func:`perceptual_divergence` watches whether the ESR ranking and the MRSTFT ranking
are drifting apart across the run log. Sustained divergence means ESR is being gamed
and needs a human to look.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Dict, Optional, Sequence

import torch as _torch

from .constants import (
    MRSTFT_FFT_SIZES as _FFT_SIZES,
    MRSTFT_HOP_SIZES as _HOP_SIZES,
    MRSTFT_WIN_LENGTHS as _WIN_LENGTHS,
    PRE_EMPH_COEF as _PRE_EMPH_COEF,
)

try:
    from nam.models.losses import (
        apply_pre_emphasis_filter as _apply_pre_emphasis_filter,
        esr as _nam_esr,
    )
    from nam._dependencies.auraloss.freq import (
        MultiResolutionSTFTLoss as _MultiResolutionSTFTLoss,
    )
except ImportError as e:  # pragma: no cover - environment problem, not logic
    raise ImportError(
        "This harness reuses NAM's own loss definitions so results stay comparable "
        "with published NAM/A2 numbers. Install the trainer:\n"
        "    pip install -e /path/to/neural-amp-modeler\n"
        f"(original error: {e})"
    ) from e

__all__ = ["CaptureMetrics", "PanelMetrics", "evaluate_capture", "aggregate", "perceptual_divergence"]


_MRSTFT_CACHE: Dict[str, _MultiResolutionSTFTLoss] = {}


def _mrstft_for(device: _torch.device) -> _MultiResolutionSTFTLoss:
    key = str(device)
    if key not in _MRSTFT_CACHE:
        _MRSTFT_CACHE[key] = _MultiResolutionSTFTLoss(
            fft_sizes=list(_FFT_SIZES),
            hop_sizes=list(_HOP_SIZES),
            win_lengths=list(_WIN_LENGTHS),
        ).to(device)
    return _MRSTFT_CACHE[key]


@_dataclass
class CaptureMetrics:
    """Metrics for a single capture."""

    name: str
    esr: float
    mrstft: float

    def as_dict(self) -> Dict[str, float]:
        return {"name": self.name, "esr": self.esr, "mrstft": self.mrstft}


@_dataclass
class PanelMetrics:
    """Aggregated metrics across the panel plus the generalization holdout."""

    esr: float
    mrstft: float
    esr_holdout: Optional[float]
    per_capture: Sequence[CaptureMetrics]

    def as_dict(self) -> Dict[str, object]:
        return {
            "esr": self.esr,
            "mrstft": self.mrstft,
            "esr_holdout": self.esr_holdout,
            "per_capture": [c.as_dict() for c in self.per_capture],
        }


def _as_2d(z: _torch.Tensor) -> _torch.Tensor:
    return z[None] if z.ndim == 1 else z


@_torch.no_grad()
def evaluate_capture(
    name: str,
    preds: _torch.Tensor,
    targets: _torch.Tensor,
    *,
    pre_emph_coef: Optional[float] = _PRE_EMPH_COEF,
) -> CaptureMetrics:
    """Score one capture's predictions against its target.

    :param preds: (N,) or (B,N) predicted samples.
    :param targets: Same shape as ``preds``.
    :param pre_emph_coef: Pre-emphasis coefficient applied before ESR. ``None``
        disables pre-emphasis (raw ESR).
    """
    preds, targets = _as_2d(preds), _as_2d(targets)
    if preds.shape != targets.shape:
        raise ValueError(
            f"preds/targets shape mismatch for capture {name!r}: "
            f"{tuple(preds.shape)} vs {tuple(targets.shape)}"
        )

    if pre_emph_coef is not None:
        esr_preds = _apply_pre_emphasis_filter(preds, pre_emph_coef)
        esr_targets = _apply_pre_emphasis_filter(targets, pre_emph_coef)
    else:
        esr_preds, esr_targets = preds, targets

    esr_value = float(_nam_esr(esr_preds, esr_targets).item())

    mrstft_fn = _mrstft_for(preds.device)
    mrstft_value = float(mrstft_fn(preds[:, None, :], targets[:, None, :]).item())

    return CaptureMetrics(name=name, esr=esr_value, mrstft=mrstft_value)


def aggregate(
    panel: Sequence[CaptureMetrics],
    holdout: Sequence[CaptureMetrics] = (),
) -> PanelMetrics:
    """Combine per-capture metrics into the headline score.

    The panel mean is the score. Scoring on a single capture would let the search
    overfit the *architecture* to one amp's nonlinearity, which is the failure mode
    that would make the whole exercise worthless.
    """
    if not panel:
        raise ValueError("Cannot aggregate an empty panel.")

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    return PanelMetrics(
        esr=mean([c.esr for c in panel]),
        mrstft=mean([c.mrstft for c in panel]),
        esr_holdout=mean([c.esr for c in holdout]) if holdout else None,
        per_capture=list(panel) + list(holdout),
    )


def perceptual_divergence(esr_series: Sequence[float], mrstft_series: Sequence[float]) -> float:
    """Spearman rank correlation between the ESR and MRSTFT orderings.

    Fed the run log's history. A value near 1.0 means the two metrics agree about
    which experiments were better. A falling or negative value means the search is
    improving ESR while MRSTFT disagrees -- the signature of metric gaming, and a
    cue that a human should listen to the outputs before trusting the log.

    Returns ``nan`` when there is too little history, or when either series is
    constant (rank correlation is undefined for a constant series).
    """
    n = len(esr_series)
    if n != len(mrstft_series):
        raise ValueError("Series length mismatch.")
    if n < 3:
        return float("nan")

    def ranks(xs: Sequence[float]) -> list[float]:
        order = sorted(range(len(xs)), key=lambda i: xs[i])
        out = [0.0] * len(xs)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
                j += 1
            shared = (i + j) / 2.0
            for k in range(i, j + 1):
                out[order[k]] = shared
            i = j + 1
        return out

    ra, rb = ranks(esr_series), ranks(mrstft_series)
    mean_a, mean_b = sum(ra) / n, sum(rb) / n
    num = sum((a - mean_a) * (b - mean_b) for a, b in zip(ra, rb))
    den_a = sum((a - mean_a) ** 2 for a in ra)
    den_b = sum((b - mean_b) ** 2 for b in rb)
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b) ** 0.5
