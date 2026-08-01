"""
Experiment driver. READ-ONLY: not to be modified by the research agent.

Owns the three things that must be identical across every experiment for the run log
to mean anything: the training time budget, the evaluation procedure, and the printed
results contract.

The agent writes its own training loop in ``train.py`` but calls :class:`TimeBudget`
to know when to stop and :func:`report` to be scored. Keeping evaluation here rather
than in ``train.py`` is the point -- an agent that could edit its own evaluation
would eventually, and gradually, edit it into something flattering.
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass as _dataclass
from typing import Iterable, List, Optional, Sequence

import torch as _torch

from .constants import EXPERIMENT_SECONDS
from .cost import CostReport, count_cost
from .data import Capture, Panel
from .metrics import CaptureMetrics, PanelMetrics, aggregate, evaluate_capture
from .reference import a2_standard
from .verdict import Budget, check_budget

__all__ = [
    "TimeBudget",
    "RunResult",
    "evaluate_models",
    "measure_cost",
    "report",
    "valid_forward",
]

#: Evaluation chunk length. Bounds peak memory on long validation segments without
#: changing the result: chunks overlap by the receptive field, so every output sample
#: sees exactly the context it would see in one pass.
_EVAL_CHUNK = 1 << 18


class TimeBudget:
    """Wall-clock training budget, shared evenly across the panel.

    Excludes startup, compilation and evaluation, so that a slow-to-compile
    architecture is not penalised for something the end user never pays.
    """

    def __init__(self, total_seconds: float = EXPERIMENT_SECONDS, n_captures: int = 1):
        if n_captures < 1:
            raise ValueError("n_captures must be >= 1")
        self.per_capture = total_seconds / n_captures
        self._start: Optional[float] = None
        self.elapsed_total = 0.0

    def start(self) -> "TimeBudget":
        self._start = _time.monotonic()
        return self

    @property
    def expired(self) -> bool:
        if self._start is None:
            raise RuntimeError("TimeBudget.start() was never called.")
        return (_time.monotonic() - self._start) >= self.per_capture

    @property
    def remaining(self) -> float:
        if self._start is None:
            return self.per_capture
        return max(0.0, self.per_capture - (_time.monotonic() - self._start))

    def finish(self) -> float:
        """Stop the clock for the current capture and bank the time."""
        if self._start is None:
            raise RuntimeError("TimeBudget.start() was never called.")
        spent = _time.monotonic() - self._start
        self.elapsed_total += spent
        self._start = None
        return spent


@_dataclass
class RunResult:
    metrics: PanelMetrics
    cost: CostReport
    budget: Budget
    training_seconds: float
    status: str  # ok | invalid
    note: str = ""


def _infer_receptive_field(model, device) -> int:
    """Determine lookback, preferring the model's own declaration.

    A declared value is preferred because it cannot be inferred for models that pad
    their input: those return an output as long as their input, which is
    indistinguishable from a receptive field of 1.
    """
    declared = getattr(model, "receptive_field", None)
    if isinstance(declared, int) and declared > 0:
        return declared

    probe = 1 << 14
    with _torch.no_grad():
        out = model(_torch.zeros(1, probe, device=device))
    out = out[0] if isinstance(out, (tuple, list)) else out
    inferred = probe - int(out.shape[-1]) + 1
    if inferred <= 1:
        raise ValueError(
            "Cannot infer receptive field: the model returns as many samples as it "
            "was given, which means it pads internally. Expose a `receptive_field` "
            "attribute so the warmup region can be excluded from training and "
            "scoring."
        )
    return inferred


def valid_forward(model, x: _torch.Tensor, receptive_field: int) -> _torch.Tensor:
    """Run ``model`` and return only outputs that saw a full context window.

    Two conventions are in the wild and both are accepted:

    * **trimming** -- returns ``L - receptive_field + 1`` samples (plain causal conv).
    * **padding** -- returns ``L`` samples, having zero-padded the start. This is
      NAM's default (``BaseNet.pad_start_default``).

    Under the padding convention the leading ``receptive_field - 1`` outputs are
    computed from zeros rather than from audio. They are dropped here. Training or
    scoring on them would be fitting fabricated signal, which flatters quiet
    passages in particular because the padded region is silent.

    Returned samples align to ``x[receptive_field - 1:]``.
    """
    out = model(x)
    out = out[0] if isinstance(out, (tuple, list)) else out

    n_in = int(x.shape[-1])
    n_out = int(out.shape[-1])
    trimmed = n_in - receptive_field + 1

    if n_out == trimmed:
        return out
    if n_out == n_in:
        return out[..., receptive_field - 1:]
    raise ValueError(
        f"Model returned {n_out} samples for {n_in} inputs; expected {trimmed} "
        f"(trimming) or {n_in} (start-padding) for receptive field {receptive_field}."
    )


@_torch.no_grad()
def _predict(model, x: _torch.Tensor, receptive_field: int) -> _torch.Tensor:
    """Run ``model`` over a long signal in overlapping chunks.

    Returns the prediction aligned to the tail of ``x``: output ``i`` corresponds to
    ``x[receptive_field - 1 + i]``.
    """
    n = int(x.numel())
    usable = n - receptive_field + 1
    if usable <= 0:
        raise ValueError(
            f"Validation segment ({n} samples) is shorter than the model's receptive "
            f"field ({receptive_field}); cannot evaluate."
        )

    pieces: List[_torch.Tensor] = []
    stride = max(1, _EVAL_CHUNK - receptive_field + 1)
    start = 0
    while start < usable:
        stop = min(start + stride + receptive_field - 1, n)
        chunk = x[start:stop][None]
        pieces.append(valid_forward(model, chunk, receptive_field).reshape(-1))
        start += stride

    return _torch.cat(pieces)[:usable]


@_torch.no_grad()
def _score_one(model, capture: Capture, device) -> CaptureMetrics:
    was_training = model.training
    model.eval()
    try:
        rf = _infer_receptive_field(model, device)
        preds = _predict(model, capture.x_val.to(device), rf)
        target = capture.y_val.to(device)[-preds.numel():]
        return evaluate_capture(capture.name, preds, target)
    finally:
        model.train(was_training)


@_torch.no_grad()
def evaluate_models(models: dict, panel: Panel, device=None) -> PanelMetrics:
    """Score one model per capture.

    NAM models are per-capture: a .nam file captures one amp at one setting. So an
    *architecture* is evaluated by training a fresh model on each capture with the
    same recipe and averaging. This is also what makes the holdout meaningful --
    holdout amps get their own freshly-trained models, because the thing being asked
    to generalize is the architecture, not a set of weights.

    :param models: Maps capture name -> trained model.
    """
    missing = [c.name for c in list(panel.panel) + list(panel.holdout) if c.name not in models]
    if missing:
        raise KeyError(f"No trained model supplied for captures: {missing}")

    def score(captures: Sequence[Capture]) -> List[CaptureMetrics]:
        out = []
        for c in captures:
            model = models[c.name]
            dev = device or next(model.parameters()).device
            out.append(_score_one(model, c, dev))
        return out

    return aggregate(score(panel.panel), score(panel.holdout))


def measure_cost(model, device: str = "cpu") -> CostReport:
    """Measure the model's streaming compute cost."""
    return count_cost(model, device=device)


def _reference_budget() -> Budget:
    """Recompute the cap from the frozen reference on every run.

    Deliberately not cached to disk: there is then no artifact to edit, and the cap
    cannot drift out of sync with the architecture it claims to represent.
    """
    return Budget.from_reference(count_cost(a2_standard(), device="cpu"))


def report(
    models: dict,
    panel: Panel,
    training_seconds: float,
    *,
    rtf: Optional[float] = None,
    device=None,
) -> RunResult:
    """Evaluate, cost, gate, and print the results contract.

    The printed block is what the research agent greps; keep it stable.

    :param models: Maps capture name -> trained model. Cost is measured on one of
        them, since every capture is trained with the same architecture.
    """
    metrics = evaluate_models(models, panel, device=device)
    representative = models[panel.panel[0].name]
    cost = measure_cost(representative, device="cpu")
    budget = _reference_budget()

    status, note = "ok", ""
    breach = check_budget(cost, budget)
    if breach is not None:
        status, note = "invalid", breach
    elif not cost.is_linear:
        status = "invalid"
        note = (
            f"cost is not linear in context length (nonlinearity "
            f"{cost.nonlinearity:.3f}); per-sample cost depends on accumulated "
            f"context, so it cannot be compared against fixed-cost models"
        )

    print("---")
    print(f"esr:              {metrics.esr:.6f}")
    print(f"mrstft:           {metrics.mrstft:.6f}")
    holdout = "n/a" if metrics.esr_holdout is None else f"{metrics.esr_holdout:.6f}"
    print(f"esr_holdout:      {holdout}")
    print(f"macs_per_sample:  {cost.macs_per_sample:.1f}")
    print(f"mac_budget:       {budget.macs_per_sample:.1f}")
    print(f"elementwise:      {cost.elementwise_per_sample:.1f}")
    print(f"params:           {cost.params}")
    print(f"rtf:              {'n/a' if rtf is None else f'{rtf:.4f}'}")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"status:           {status}")
    if note:
        print(f"note:             {note}")
    print("--- per capture ---")
    for c in metrics.per_capture:
        print(f"{c.name:28s} esr={c.esr:.6f}  mrstft={c.mrstft:.6f}")

    return RunResult(
        metrics=metrics,
        cost=cost,
        budget=budget,
        training_seconds=training_seconds,
        status=status,
        note=note,
    )
