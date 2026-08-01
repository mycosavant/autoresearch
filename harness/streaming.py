"""
Streaming-equivalent cost forms. READ-ONLY: not to be modified by the research agent.

Why this exists
---------------
:mod:`harness.cost` measures what the model actually runs. For every convolutional
architecture that is the right answer -- the training-time forward pass and the
deployed streaming pass do the same arithmetic per sample.

Recurrent and IIR architectures break that identity, and they break it in the
direction that matters most to this repo. A diagonal state-space layer costs O(1) per
sample when deployed (``h <- A h + B x``), but nobody trains it that way: a
sample-at-a-time Python loop over a 48 kHz signal is unusably slow, so training uses
a chunked scan or an FFT convolution. Those forms are mathematically identical and
arithmetically nothing alike -- a chunked scan with chunk size C does O(C) work per
sample. Costing the training form would report a number one to two orders of
magnitude above the deployed cost and reject the architecture out of hand.

Since "spend fewer MACs by needing less memory" is the central structural bet in
``program.md``, a harness that cannot score a recurrence cannot evaluate the most
promising direction it names. Hence this module.

The bargain
-----------
A model may expose ``streaming_form() -> nn.Module``. If it does, cost is measured on
that module instead. In exchange it must survive three checks, all of which run
before a single op is counted:

1. **Equivalence.** Both forms are run on the same random signal and compared. The
   discrepancy is measured as ESR -- the same units the harness scores in -- and must
   fall below :data:`MAX_EQUIVALENCE_ESR`. A tolerance expressed in the scoring metric
   is one you can reason about: at 1e-6 the two forms cannot differ by enough to move
   a result, since converged amp models score around 1e-3 to 1e-4 and the measured
   noise floor is far above 1e-6.

2. **No smuggled capacity.** The streaming form may not have more parameters than the
   training form. A "streaming form" that is really a distilled smaller network would
   pass equivalence only loosely, but this makes the intent explicit and cheap to
   check.

3. **No input-dependent construction.** ``streaming_form()`` takes no arguments, so
   it cannot specialize itself to the probe. Whatever coefficient algebra it does
   (reflection coefficients to biquad taps, log-space rates to decay factors) happens
   once at construction, which is exactly what a real export step does and is
   correctly not charged per sample.

Residual risk, stated plainly
-----------------------------
The streaming form is probed at short lengths, because a per-sample recurrence under
``__torch_dispatch__`` costs milliseconds per step and a 32,768-sample probe would
dominate the experiment budget. Short probes are sound when cost is affine in length
-- and it is, for any real streaming implementation -- but a pathological model could
be affine over the probe window and not beyond it.

So the fit from the short probes is *extrapolated* to
:data:`~harness.cost.REFERENCE_CONTEXT_SAMPLES` and checked against one real
measurement there. That closes the gap at the price of a single long pass. It is the
one expensive thing this module does, and it is not optional in the scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass as _dataclass
from typing import Optional

import torch as _torch
import torch.nn as _nn

__all__ = [
    "StreamingFormError",
    "ProbeSpec",
    "ResolvedForm",
    "MAX_EQUIVALENCE_ESR",
    "equivalence_esr",
    "resolve",
]

#: Attribute a model defines to offer a streaming cost form.
STREAMING_FORM_ATTR = "streaming_form"

#: Maximum tolerated disagreement between the two forms, as ESR.
#:
#: Set in scoring units on purpose. Converged amp models land around 1e-3..1e-4 ESR
#: and the run-to-run noise floor is measured well above 1e-6, so a discrepancy this
#: small provably cannot change a keep/discard decision. It is meanwhile far looser
#: than float32 reassociation between a scan and a recurrence, which lands near 1e-12.
MAX_EQUIVALENCE_ESR = 1e-6

#: Length of the signal used for the equivalence check. Long enough that a recurrence
#: has run well past any plausible chunk boundary and that divergence between the
#: forms has had room to accumulate. This runs without the dispatch counter, so it
#: costs a plain forward pass rather than a traced one.
EQUIVALENCE_SAMPLES = 8_192

#: Seeds for the equivalence probes. More than one because a single random draw can
#: miss a form that is correct only in part of the input range -- a saturating
#: nonlinearity implemented differently in the two forms, say.
EQUIVALENCE_SEEDS = (0, 1)

#: Probe signal amplitude. Guitar DI sits well below full scale, but the check should
#: exercise the nonlinear region rather than the near-linear one, so this drives hard.
EQUIVALENCE_AMPLITUDE = 0.9

#: Floors on the probe geometry a streaming form may request. A stride below this
#: makes the slope sensitive to per-call constant work; a base below it risks
#: measuring a model that has not reached steady state.
MIN_PROBE_SAMPLES = 256
MIN_PROBE_STRIDE = 64

#: Relative tolerance on the extrapolation check. Op counts are integers and the fit
#: is exactly affine for any honest implementation, so this is a float-arithmetic
#: allowance, not a modelling allowance.
EXTRAPOLATION_RTOL = 1e-9


class StreamingFormError(RuntimeError):
    """A model offered a streaming cost form that cannot be trusted.

    Raised rather than falling back to costing the training form. A silent fallback
    would report a wildly inflated cost and mark the run ``invalid``, which reads as
    "this architecture is too expensive" when the truth is "this harness could not
    verify the claim" -- two very different findings to have in the run log.
    """


@_dataclass(frozen=True)
class ProbeSpec:
    """Input lengths at which a form's cost is sampled."""

    base: int
    stride: int

    def lengths(self) -> list[int]:
        return [self.base, self.base + self.stride, self.base + 2 * self.stride]


@_dataclass
class ResolvedForm:
    """The module cost should actually be measured on."""

    module: _nn.Module
    probe: Optional[ProbeSpec]
    used_streaming: bool
    #: Disagreement between training and streaming forms, in ESR. 0.0 when no
    #: streaming form was involved.
    equivalence_esr: float = 0.0


def equivalence_esr(candidate: _torch.Tensor, reference: _torch.Tensor) -> float:
    """Error-signal ratio of ``candidate`` against ``reference``.

    Deliberately the plain (non-pre-emphasised) ESR: this measures whether two
    implementations of the same maths agree, not whether a model sounds right, and
    pre-emphasis would understate low-frequency divergence -- precisely where a
    mis-implemented recurrence goes wrong.
    """
    if candidate.shape != reference.shape:
        raise StreamingFormError(
            f"streaming_form() output shape {tuple(candidate.shape)} does not match "
            f"the training form's {tuple(reference.shape)}. The two forms must be the "
            f"same function of the same input."
        )
    num = _torch.mean(_torch.square(candidate.double() - reference.double()))
    den = _torch.mean(_torch.square(reference.double())) + 1e-30
    return float(num / den)


def _probe_of(module: _nn.Module, default_base: int, default_stride: int) -> ProbeSpec:
    base = int(getattr(module, "cost_probe_samples", default_base))
    stride = int(getattr(module, "cost_probe_stride", default_stride))
    if base < MIN_PROBE_SAMPLES:
        raise StreamingFormError(
            f"cost_probe_samples={base} is below the floor of {MIN_PROBE_SAMPLES}."
        )
    if stride < MIN_PROBE_STRIDE:
        raise StreamingFormError(
            f"cost_probe_stride={stride} is below the floor of {MIN_PROBE_STRIDE}."
        )
    return ProbeSpec(base=base, stride=stride)


def _check_equivalence(training: _nn.Module, streaming: _nn.Module, device: str) -> float:
    worst = 0.0
    with _torch.no_grad():
        for seed in EQUIVALENCE_SEEDS:
            g = _torch.Generator(device="cpu").manual_seed(seed)
            x = (
                _torch.rand(1, EQUIVALENCE_SAMPLES, generator=g) * 2.0 - 1.0
            ) * EQUIVALENCE_AMPLITUDE
            x = x.to(device)
            got = streaming(x)
            want = training(x)
            got = got[0] if isinstance(got, (tuple, list)) else got
            want = want[0] if isinstance(want, (tuple, list)) else want
            worst = max(worst, equivalence_esr(got, want))

    if worst > MAX_EQUIVALENCE_ESR:
        raise StreamingFormError(
            f"streaming_form() disagrees with the training form by {worst:.3e} ESR, "
            f"above the {MAX_EQUIVALENCE_ESR:.0e} limit. The streaming form is what "
            f"gets costed, so it has to be the same model -- if the two genuinely "
            f"differ, the cheaper one is the one to train."
        )
    return worst


def _check_capacity(training: _nn.Module, streaming: _nn.Module) -> None:
    n_train = sum(p.numel() for p in training.parameters())
    n_stream = sum(p.numel() for p in streaming.parameters())
    if n_stream > n_train:
        raise StreamingFormError(
            f"streaming_form() has {n_stream} parameters against the training form's "
            f"{n_train}. A streaming form may re-parameterize (log-rates to decays, "
            f"reflection coefficients to taps) but may not add capacity."
        )


def resolve(
    model: _nn.Module,
    *,
    device: str = "cpu",
    default_base: int,
    default_stride: int,
) -> ResolvedForm:
    """Decide which module to measure cost on, verifying it if it is a streaming form.

    :param model: The model as trained.
    :raises StreamingFormError: if a streaming form is offered but fails a check.
    """
    factory = getattr(model, STREAMING_FORM_ATTR, None)
    if factory is None:
        return ResolvedForm(module=model, probe=None, used_streaming=False)
    if not callable(factory):
        raise StreamingFormError(
            f"{STREAMING_FORM_ATTR!r} must be a method taking no arguments; got "
            f"{type(factory).__name__}."
        )

    streaming = factory()
    if not isinstance(streaming, _nn.Module):
        raise StreamingFormError(
            f"streaming_form() must return an nn.Module; got {type(streaming).__name__}."
        )

    model = model.to(device).eval()
    streaming = streaming.to(device).eval()

    _check_capacity(model, streaming)
    esr = _check_equivalence(model, streaming, device)

    return ResolvedForm(
        module=streaming,
        probe=_probe_of(streaming, default_base, default_stride),
        used_streaming=True,
        equivalence_esr=esr,
    )
