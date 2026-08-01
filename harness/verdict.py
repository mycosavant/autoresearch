"""
The keep/discard decision. READ-ONLY: not to be modified by the research agent.

Two guardrails live here, and they are the difference between a research log and a
random walk.

**The compute gate.** Score is only meaningful under a fixed compute cap, so a run
over budget is ``INVALID`` rather than merely worse. Without this the agent buys
accuracy with compute and every "improvement" is uninterpretable.

**The noise floor.** ESR varies run to run for reasons that have nothing to do with
the architecture (init, data order, nondeterministic kernels). A loop that keeps any
improvement will bank that variance dozens of times and drift somewhere meaningless
while the log shows steady progress. So an improvement must clear the measured noise
floor before it is kept. This is the single most important difference from the
upstream LLM loop, where the metric is stable enough for the naive rule to work.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from enum import Enum as _Enum
from typing import TYPE_CHECKING, Optional, Sequence

from .constants import (
    ELEMENTWISE_BUDGET_FACTOR as _EW_FACTOR,
    KEEP_MARGIN_FACTOR as _KEEP_MARGIN,
    MAC_BUDGET_TOLERANCE as _MAC_TOL,
)
from .cost import CostReport

if TYPE_CHECKING:  # pragma: no cover
    from .cost_model import CostModel

__all__ = ["Verdict", "Budget", "Decision", "check_budget", "noise_floor", "decide"]


class Verdict(str, _Enum):
    KEEP = "keep"
    DISCARD = "discard"
    CRASH = "crash"
    INVALID = "invalid"


@_dataclass(frozen=True)
class Budget:
    """The compute cap, derived from the frozen reference architecture.

    Deliberately not loaded from a file: the cap is recomputed from
    ``harness/reference.py`` on every run, so there is no artifact to tamper with
    and no way for the cap to drift out of sync with what it claims to represent.

    Two layers. The MAC and elementwise caps are cheap necessary conditions, always
    enforced. The predicted-runtime cap is the one that actually reflects hardware --
    but it only binds once the cost model is adequately calibrated, because gating on
    an unidentified model would be worse than gating on raw MACs: it would be just as
    wrong and harder to see through.
    """

    macs_per_sample: float
    elementwise_per_sample: float
    predicted_us_per_sample: Optional[float] = None
    model_gates: bool = False

    @classmethod
    def from_reference(
        cls, report: CostReport, cost_model: "CostModel | None" = None
    ) -> "Budget":
        predicted = None
        gates = False
        if cost_model is not None:
            from .cost_model import features_of

            predicted = cost_model.predict(features_of(report)) * (1.0 + _MAC_TOL)
            gates = cost_model.is_adequate
        return cls(
            macs_per_sample=report.macs_per_sample * (1.0 + _MAC_TOL),
            elementwise_per_sample=report.elementwise_per_sample * _EW_FACTOR,
            predicted_us_per_sample=predicted,
            model_gates=gates,
        )


@_dataclass
class Decision:
    verdict: Verdict
    reason: str

    @property
    def is_keep(self) -> bool:
        return self.verdict is Verdict.KEEP


def check_budget(
    report: CostReport, budget: Budget, cost_model: "CostModel | None" = None
) -> Optional[str]:
    """Return a rejection reason if ``report`` breaches the cap, else ``None``."""
    if report.macs_per_sample > budget.macs_per_sample:
        over = report.macs_per_sample / budget.macs_per_sample - 1.0
        return (
            f"over MAC budget by {over:.1%} "
            f"({report.macs_per_sample:.0f} > {budget.macs_per_sample:.0f} MACs/sample)"
        )
    if report.elementwise_per_sample > budget.elementwise_per_sample:
        over = report.elementwise_per_sample / budget.elementwise_per_sample - 1.0
        return (
            f"over elementwise budget by {over:.1%} "
            f"({report.elementwise_per_sample:.1f} > "
            f"{budget.elementwise_per_sample:.1f} ops/sample)"
        )

    # Predicted runtime binds only once the model can carry the weight.
    if budget.model_gates and cost_model is not None and budget.predicted_us_per_sample:
        from .cost_model import features_of

        predicted = cost_model.predict(features_of(report))
        if predicted > budget.predicted_us_per_sample:
            over = predicted / budget.predicted_us_per_sample - 1.0
            return (
                f"over predicted-runtime budget by {over:.1%} "
                f"({predicted:.4f} > {budget.predicted_us_per_sample:.4f} us/sample)"
            )
    return None


def noise_floor(repeat_scores: Sequence[float]) -> float:
    """Estimate the ESR noise floor from repeated runs of an identical config.

    Uses the full spread (max - min) rather than a standard deviation: with only a
    handful of seeds, spread is the honest statement of "differences this small are
    not evidence of anything", and it does not pretend to a distributional
    assumption the sample size cannot support.
    """
    if len(repeat_scores) < 2:
        raise ValueError(
            f"Need at least 2 repeats to estimate a noise floor, got {len(repeat_scores)}."
        )
    return max(repeat_scores) - min(repeat_scores)


def decide(
    candidate_esr: float,
    incumbent_esr: Optional[float],
    cost: CostReport,
    budget: Budget,
    floor: float,
    *,
    candidate_holdout: Optional[float] = None,
    incumbent_holdout: Optional[float] = None,
) -> Decision:
    """Decide the fate of one experiment.

    :param candidate_esr: Panel-mean ESR of the new candidate.
    :param incumbent_esr: Panel-mean ESR of the current branch tip. ``None`` for the
        very first (baseline) run, which is always kept.
    :param floor: Measured noise floor from :func:`noise_floor`.
    :param candidate_holdout: Holdout ESR, used only to flag suspicious wins -- it
        never drives the decision, because optimizing against the holdout would
        destroy the thing it exists to measure.
    """
    breach = check_budget(cost, budget)
    if breach is not None:
        return Decision(Verdict.INVALID, breach)

    if incumbent_esr is None:
        return Decision(Verdict.KEEP, "baseline established")

    delta = incumbent_esr - candidate_esr  # positive means candidate is better
    required = floor * _KEEP_MARGIN

    if delta <= 0:
        return Decision(
            Verdict.DISCARD,
            f"no improvement (esr {candidate_esr:.6f} vs incumbent {incumbent_esr:.6f})",
        )

    if delta < required:
        return Decision(
            Verdict.DISCARD,
            f"improvement {delta:.6f} within noise floor {required:.6f}; not evidence",
        )

    reason = f"improved esr by {delta:.6f} (noise floor {required:.6f})"

    # Report, do not veto: a real architectural win can legitimately shift holdout
    # performance, and auto-rejecting on it would turn the holdout into another
    # training signal.
    if candidate_holdout is not None and incumbent_holdout is not None:
        if candidate_holdout > incumbent_holdout:
            reason += (
                f" -- WARNING: holdout worsened "
                f"({incumbent_holdout:.6f} -> {candidate_holdout:.6f}); "
                f"possible panel overfit"
            )

    return Decision(Verdict.KEEP, reason)
