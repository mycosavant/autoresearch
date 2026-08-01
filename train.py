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

import math

import torch
import torch.nn as nn
import torch.nn.functional as _nnf

from harness.constants import EXPERIMENT_SECONDS, PRE_EMPH_COEF
from harness.data import load_panel
from harness.reference import (
    A2_RECEPTIVE_FIELD,
    a1_standard,
    a2_nano,
    a2_standard,
    lstm_baseline,
)
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
#:   wh_ssm        10,160 MACs/sample, 10,161 params, rf 6,347  <- candidate, untrained
BASELINE = "a2_standard"


# --------------------------------------------------------------------------------
# wh_ssm knobs
# --------------------------------------------------------------------------------

#: Channels carried between stages of the cascade.
WH_CHANNELS = 20

#: Resonant blocks per state-space stage. Each is a damped 2-D rotation, i.e. a pair
#: of conjugate poles, i.e. one second-order section. So this is 2 states per block.
WH_BLOCKS = 40

#: Hidden width of each static nonlinearity.
WH_NL_HIDDEN = 40

#: Pole time constants at initialization, in samples. Spread log-uniformly so the
#: bank starts with memory at every scale from a few samples (bright transient
#: detail) up to a few thousand (cabinet ring, sag).
WH_TAU_RANGE = (4.0, 4000.0)

#: Pole frequencies at initialization, in Hz. Guitar-relevant band.
WH_FREQ_RANGE = (60.0, 8000.0)

#: Hard ceiling on pole radius. Not cosmetic: ``sigmoid`` alone does NOT keep r below
#: 1 in float32 -- it saturates to exactly 1.0 around a logit of 17, which puts a pole
#: on the unit circle (a pure integrator) and makes the ``sqrt(1 - r^2)`` drive gain
#: zero with an infinite derivative, so the first gradient step through it is NaN.
#: Scaling by a constant strictly below 1 restores the guarantee the reparameterization
#: was supposed to provide. 0.99999 is a time constant of 100k samples, ~2 s at 48 kHz,
#: which is longer than any amp behaviour this is meant to capture.
WH_R_MAX = 0.99999


# ================================================================================
# Model
# ================================================================================


class ResonantBank(nn.Module):
    """A bank of damped 2-D rotations: parallel second-order sections, learned.

    Each block holds a 2-D state driven by a rotation-scaling matrix
    ``A_i = r_i R(theta_i)``, which is exactly a conjugate pole pair -- a resonance
    at ``theta_i`` decaying with time constant ``-1/log r_i``. A bank of them, mixed
    linearly, spans the same space as a parallel biquad filter bank.

    Why this shape. A guitar amp's linear blocks are resonant and long: transformer
    and cabinet response, coupling networks, supply sag. A dilated conv stack pays a
    tap per sample of memory it wants; a pole pays 4 MACs for memory that never ends.
    A2 spends 11,776 MACs/sample to reach 132 ms of context. One block here reaches
    further than that for 4.

    Stability is structural, not a penalty term: ``r = WH_R_MAX * sigmoid(.)`` is
    strictly below 1 for every value the optimizer can reach, so no parameter setting
    produces a divergent pole. (The ``WH_R_MAX`` factor is load-bearing -- ``sigmoid``
    alone saturates to exactly 1.0 in float32. See that constant.)

    Two forms, one function:

    * ``_scan`` -- what trains. A doubling scan over the sequence: exact, parallel
      over time, ``ceil(log2 T)`` sequential rounds instead of T.
    * ``_recurrent`` -- what deploys, and what gets costed. Everything pointwise in
      time stays vectorized; only the 2-state update loops. O(1) MACs per sample.

    They are the same arithmetic in a different association order, so they agree to
    float noise -- which is what ``harness.streaming`` checks before believing the
    cheaper one.
    """

    def __init__(self, in_ch: int, out_ch: int, n_blocks: int):
        super().__init__()
        self.in_ch, self.out_ch, self.n_blocks = in_ch, out_ch, n_blocks

        self.b_proj = nn.Parameter(torch.randn(n_blocks, 2, in_ch) / math.sqrt(in_ch))

        tau = torch.logspace(
            math.log10(WH_TAU_RANGE[0]), math.log10(WH_TAU_RANGE[1]), n_blocks
        )
        frac = torch.exp(-1.0 / tau) / WH_R_MAX
        self.r_logit = nn.Parameter(torch.log(frac / (1.0 - frac)))

        freq = torch.logspace(
            math.log10(WH_FREQ_RANGE[0]), math.log10(WH_FREQ_RANGE[1]), n_blocks
        )
        self.theta = nn.Parameter(2.0 * math.pi * freq / 48_000.0)

        self.c_proj = nn.Linear(2 * n_blocks, out_ch)

    # -- shared pieces ----------------------------------------------------------

    def _radius(self) -> torch.Tensor:
        """Pole radii, strictly inside the unit circle. See :data:`WH_R_MAX`."""
        return WH_R_MAX * torch.sigmoid(self.r_logit)

    def _drive(self, x: torch.Tensor) -> torch.Tensor:
        """(B, in_ch, T) -> (B, n_blocks, 2, T). Pointwise in time, so identical in
        both forms and correctly charged ``2 * n_blocks * in_ch`` MACs per sample.

        The drive is scaled by ``sqrt(1 - r^2)``, which gives every block unit energy
        gain against white input regardless of its pole radius. Without it, radius and
        gain are the same knob: a pole at radius r has resonant gain of order
        ``1 / (1 - r)``, so a bank spanning time constants from 4 to 4000 samples
        spans four orders of magnitude of gain, the untrained cascade puts out ~40x
        its input, and the optimizer can only lengthen a time constant by paying for
        the amplitude blowup that comes with it. Normalizing decouples the two, which
        is what makes the bank trainable rather than merely stable.

        ``1 - r`` would normalize peak gain instead of energy, and undershoots badly:
        the resulting cascade attenuates by ~300x at initialization, so the static
        nonlinearities sit in their linear region and never engage.

        Folded into the weight rather than applied to the signal: the rescale is a
        function of parameters alone, so it happens once per call on an
        ``(n_blocks, 2, in_ch)`` tensor instead of once per sample.
        """
        r = self._radius()
        gain = torch.sqrt(1.0 - r * r)[:, None, None]
        return torch.einsum("nic,bct->bnit", self.b_proj * gain, x)

    def _read(self, h: torch.Tensor) -> torch.Tensor:
        """(B, n_blocks, 2, T) -> (B, out_ch, T)."""
        b, n, _, t = h.shape
        flat = h.reshape(b, n * 2, t).transpose(1, 2)
        return self.c_proj(flat).transpose(1, 2)

    # -- training form ----------------------------------------------------------

    def _scan(self, x: torch.Tensor) -> torch.Tensor:
        """Doubling scan. Exact, parallel over time, O(log T) sequential rounds.

        The recurrence is time-invariant, which collapses a general associative scan
        into something much cheaper: if ``h_k[t]`` accumulates the last ``2^k`` input
        terms, then ``h_{k+1}[t] = h_k[t] + A^(2^k) h_k[t - 2^k]`` -- and ``A^(2^k)``
        is one 2x2 matrix for the whole sequence, obtained by squaring the previous
        round's. After ``ceil(log2 T)`` rounds every term is included, exactly.

        This replaced a chunked scan that materialized the intra-chunk impulse
        response as a (chunk x chunk) operator. That version was memory-bound rather
        than compute-bound -- a 2 MB operator re-read 32 times per forward pass ran at
        about 50 MFLOPS -- and did ``4 * chunk`` MACs per sample against the 4 per
        round here. At a 14.5k-sample training window that is 1,024 versus 56, and
        measured ~10x end to end.

        Written as explicit multiplies rather than a 2x2 einsum because it is 2.2x
        faster and nothing here is costed: the compute cap is measured on
        ``_recurrent`` alone, so the training form is free to be shaped by whatever
        the hardware likes.
        """
        u = self._drive(x)
        h0, h1 = u[:, :, 0], u[:, :, 1]  # (B, n_blocks, T)
        a = self._state_matrix()
        step = 1
        while step < h0.shape[-1]:
            s0 = _nnf.pad(h0[..., :-step], (step, 0))
            s1 = _nnf.pad(h1[..., :-step], (step, 0))
            a00, a01 = a[:, 0, 0, None], a[:, 0, 1, None]
            a10, a11 = a[:, 1, 0, None], a[:, 1, 1, None]
            h0 = h0 + a00 * s0 + a01 * s1
            h1 = h1 + a10 * s0 + a11 * s1
            a = a @ a
            step *= 2
        return self._read(torch.stack([h0, h1], dim=2))

    # -- streaming form ---------------------------------------------------------

    def _state_matrix(self) -> torch.Tensor:
        r = self._radius()
        cos, sin = torch.cos(self.theta), torch.sin(self.theta)
        row0 = torch.stack([r * cos, -r * sin], dim=-1)
        row1 = torch.stack([r * sin, r * cos], dim=-1)
        return torch.stack([row0, row1], dim=-2)  # (n_blocks, 2, 2)

    def _recurrent(self, x: torch.Tensor) -> torch.Tensor:
        u = self._drive(x)
        a = self._state_matrix()
        h = u.new_zeros(u.shape[0], u.shape[1], 2)
        steps = []
        for t in range(u.shape[-1]):
            # A 2x2 matvec per block: 4 MACs, charged as MACs rather than dropped
            # into the elementwise bucket, which is what writing it out as scalar
            # multiplies would have done.
            h = torch.einsum("nij,bnj->bni", a, h) + u[..., t]
            steps.append(h)
        return self._read(torch.stack(steps, dim=-1))

    def forward(self, x: torch.Tensor, streaming: bool = False) -> torch.Tensor:
        return self._recurrent(x) if streaming else self._scan(x)


class StaticNonlinearity(nn.Module):
    """The 'H' of Wiener-Hammerstein: memoryless, applied per sample.

    Pointwise in time, so it needs no streaming form -- it already is one.
    """

    def __init__(self, channels: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.fc2(torch.tanh(self.fc1(z)))
        return z.transpose(1, 2)


class WienerHammersteinSSM(nn.Module):
    """Linear resonant memory, static nonlinearity, repeated. The structural bet.

    ``filter -> nonlinearity -> filter -> nonlinearity -> filter``, which is a
    Wiener-Hammerstein cascade with one extra stage, written as an explicit
    architectural prior rather than left for a generic stack to rediscover.

    ``receptive_field`` is nominal. The recurrence has unbounded memory, so there is
    no length past which the model provably ignores the input; the figure here is a
    settling window, and it is set to A2's 6,347 deliberately -- the harness drops
    exactly that many warmup samples from both, so the two are scored on identical
    supervised regions and identical effective context.
    """

    def __init__(self):
        super().__init__()
        ch, nb, hid = WH_CHANNELS, WH_BLOCKS, WH_NL_HIDDEN
        self.stage1 = ResonantBank(1, ch, nb)
        self.nl1 = StaticNonlinearity(ch, hid)
        self.stage2 = ResonantBank(ch, ch, nb)
        self.nl2 = StaticNonlinearity(ch, hid)
        self.stage3 = ResonantBank(ch, 1, nb)
        self.receptive_field = A2_RECEPTIVE_FIELD
        self._calibrate_gain()

    @torch.no_grad()
    def _calibrate_gain(self) -> None:
        """Rescale each stage's output projection so the cascade starts at unit gain.

        Default initialization leaves this cascade at about 0.28x, and captures are
        RMS-normalized on both sides, so an uncalibrated model spends its first few
        hundred optimizer steps discovering that it should be louder. Under Adam that
        is a fixed number of steps regardless of how wrong the gain is -- the update
        size is set by the learning rate, not the gradient -- so at 150 s per capture
        it is a straight subtraction from the budget that buys nothing.

        Measured rather than derived: the analytic unit-gain factor assumes the block
        states are independent and unit-variance, and they are neither. One probe pass
        per stage costs microseconds at construction and stays correct if the widths
        are changed, which a hard-coded constant would not.

        Only the output projections move. ``fc1`` inside each nonlinearity is left
        alone deliberately: it sets how hard the tanh is driven, which is a modelling
        choice, not a scaling artifact.
        """
        g = torch.Generator().manual_seed(0)
        z = (torch.rand(1, 1, 4096, generator=g) * 2.0 - 1.0) * 0.35

        for stage in (self.stage1, self.nl1, self.stage2, self.nl2, self.stage3):
            run = (
                (lambda t: stage(t, streaming=False))
                if isinstance(stage, ResonantBank)
                else stage
            )
            out = run(z)
            scale = z.std() / out.std().clamp_min(1e-12)
            proj = stage.c_proj if isinstance(stage, ResonantBank) else stage.fc2
            proj.weight.mul_(scale)
            proj.bias.mul_(scale)
            z = run(z)

    def _run(self, x: torch.Tensor, streaming: bool) -> torch.Tensor:
        z = x.unsqueeze(1)
        z = self.stage1(z, streaming=streaming)
        z = self.nl1(z)
        z = self.stage2(z, streaming=streaming)
        z = self.nl2(z)
        z = self.stage3(z, streaming=streaming)
        return z.squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._run(x, streaming=False)

    def streaming_form(self) -> nn.Module:
        return _StreamingWH(self)


class _StreamingWH(nn.Module):
    """Deployment-shaped view of :class:`WienerHammersteinSSM`, for cost only.

    Holds the same module, hence literally the same parameter tensors -- there is no
    copy that could drift from what was trained.
    """

    #: Cost is affine in length, so a short probe measures the same slope as a long
    #: one at a fraction of the dispatch overhead. The harness does not take that on
    #: trust: it extrapolates this fit to the reference length and checks it.
    cost_probe_samples = 512
    cost_probe_stride = 128

    def __init__(self, inner: WienerHammersteinSSM):
        super().__init__()
        self.inner = inner
        self.receptive_field = inner.receptive_field

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner._run(x, streaming=True)


_BASELINES = {
    "a2_standard": a2_standard,
    "a2_nano": a2_nano,
    "a1_standard": a1_standard,
    "lstm": lstm_baseline,
    "wh_ssm": WienerHammersteinSSM,
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
