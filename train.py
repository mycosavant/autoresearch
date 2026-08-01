"""
train.py -- THE FILE YOU EDIT.

Everything here is fair game: architecture, optimizer, schedule, batching,
augmentation, initialization, and the loss used for training. The harness in
``harness/`` decides how you are scored and is off limits.

Baseline as shipped: A2 standard, one model per capture, trained under a fixed
wall-clock budget split evenly across the panel and holdout.

Run:  uv run train.py > run.log 2>&1
"""

from __future__ import annotations

import torch
import torch.nn as nn

from harness.constants import EXPERIMENT_SECONDS, PRE_EMPH_COEF
from harness.data import load_panel
from harness.reference import a1_standard, a2_nano, a2_standard, lstm_baseline
from harness.runner import TimeBudget, report, valid_forward

# ================================================================================
# Knobs
# ================================================================================

SEED = 0

#: Output samples per training example. Larger amortizes the receptive-field warmup
#: over more supervised samples, so it is really a compute-efficiency knob.
NY = 8192

BATCH_SIZE = 16
LEARNING_RATE = 4e-3
LR_GAMMA = 0.993          # ExponentialLR
LR_STEP_EVERY = 100       # optimizer steps between scheduler steps

#: Train against the metric being scored. Wright et al. optimize ESR directly; NAM's
#: own trainer defaults to MSE. Worth testing which actually wins here -- they are
#: not the same objective, since ESR normalizes each window by its target energy and
#: therefore weights quiet passages far more heavily.
TRAIN_LOSS = "esr"        # "esr" | "mse"

#: Which reference architecture to train. Used for the baseline phase; once you start
#: inventing, replace build_model() outright and ignore this.
#:
#:   a2_standard  11,776 MACs/sample, 12,145 params, rf 6,347   <- the number to beat
#:   a2_nano       1,731 MACs/sample,  1,870 params, rf 6,347
#:   a1_standard  13,320 MACs/sample, 13,801 params, rf 4,093   <- more cost, less context
#:   lstm             51 MACs/sample,     82 params             <- recurrent, tiny
BASELINE = "a2_standard"


# ================================================================================
# Model
# ================================================================================


_BASELINES = {
    "a2_standard": a2_standard,
    "a2_nano": a2_nano,
    "a1_standard": a1_standard,
    "lstm": lstm_baseline,
}


def build_model() -> nn.Module:
    """Return a fresh model. **Replace this to change architecture.**

    Must accept ``(batch, samples)`` and return either
    ``(batch, samples - receptive_field + 1)`` or ``(batch, samples)`` if the model
    pads its own start; the harness handles both and drops the warmup region.
    """
    try:
        return _BASELINES[BASELINE]()
    except KeyError:
        raise ValueError(
            f"Unknown BASELINE {BASELINE!r}; expected one of {sorted(_BASELINES)}"
        ) from None


# ================================================================================
# Training
# ================================================================================


def _pre_emphasis(z: torch.Tensor, coef: float) -> torch.Tensor:
    return z[..., 1:] - coef * z[..., :-1]


def _esr(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    num = torch.mean(torch.square(preds - targets), dim=1)
    den = torch.mean(torch.square(targets), dim=1) + 1e-12
    return torch.mean(num / den)


def loss_fn(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if TRAIN_LOSS == "mse":
        return torch.mean(torch.square(preds - targets))
    return _esr(_pre_emphasis(preds, PRE_EMPH_COEF), _pre_emphasis(targets, PRE_EMPH_COEF))


def sample_batch(x: torch.Tensor, y: torch.Tensor, receptive_field: int, generator):
    """Draw a batch of random aligned windows."""
    window = receptive_field - 1 + NY
    high = int(x.numel()) - window
    if high <= 0:
        raise ValueError(
            f"Capture is shorter ({x.numel()} samples) than one training window "
            f"({window}). Reduce NY or use longer captures."
        )
    idx = torch.randint(0, high, (BATCH_SIZE,), generator=generator, device=x.device)
    offsets = torch.arange(window, device=x.device)
    xb = x[idx[:, None] + offsets]                           # (B, window)
    yb = y[idx[:, None] + offsets][:, receptive_field - 1:]  # (B, NY)
    return xb, yb


def train_one(capture, budget: TimeBudget, device) -> nn.Module:
    """Train a fresh model on one capture until its slice of the budget is spent."""
    model = build_model().to(device)
    model.train()

    receptive_field = model.receptive_field
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

    generator = torch.Generator(device=device).manual_seed(SEED)
    x, y = capture.x_train.to(device), capture.y_train.to(device)

    steps, loss = 0, torch.tensor(float("nan"))
    budget.start()
    while not budget.expired:
        xb, yb = sample_batch(x, y, receptive_field, generator)
        loss = loss_fn(valid_forward(model, xb, receptive_field), yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        steps += 1
        if steps % LR_STEP_EVERY == 0:
            scheduler.step()

    spent = budget.finish()
    print(f"[{capture.name:28s}] steps={steps:6d} loss={loss.item():.6f} ({spent:.1f}s)")
    return model


def main() -> None:
    torch.manual_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    panel = load_panel()
    print(panel)

    # The holdout is trained too, with the same recipe: what is being asked to
    # generalize is the architecture, not a set of weights. Its score is reported but
    # never drives the keep/discard decision.
    captures = list(panel.panel) + list(panel.holdout)
    budget = TimeBudget(EXPERIMENT_SECONDS, n_captures=len(captures))

    models = {c.name: train_one(c, budget, device) for c in captures}

    report(models, panel, training_seconds=budget.elapsed_total, device=device)


if __name__ == "__main__":
    main()
