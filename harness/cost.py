"""
Compute-cost accounting. READ-ONLY: not to be modified by the research agent.

The research loop scores accuracy *under a fixed compute cap*, so the cost number is
as much "ground truth" as the error metric is. That makes it worth some paranoia.

Design notes
------------
Cost is measured by intercepting every ATen operation via ``__torch_dispatch__``
rather than by walking ``nn.Module`` children or registering forward hooks. Hooks
only see module boundaries, so a model that calls ``F.conv1d``/``torch.matmul``
directly -- or that hides work inside a custom autograd Function -- would report
zero cost while doing real work. Dispatch sees the actual ops regardless of how
they were spelled.

Ops fall into three buckets:

* ``_MAC_RULES``   -- multiply-accumulate work (convolutions, matmuls). Counted.
* ``_ELEMENTWISE`` -- pointwise math. No MACs, but not free on real hardware, so
  tracked separately as a second budget. An architecture that moves its work into
  giant elementwise gating would otherwise look free.
* ``_FREE``        -- views, reshapes, metadata. Genuinely ~zero.

Anything else raises :class:`UnregisteredOpError`, which marks the run invalid.
Defaulting unknown ops to zero would make the cap silently forgeable, which is the
one failure mode that would invalidate every result the loop produces.

How cost is amortized
---------------------
Cost is reported per *output audio sample* in the streaming sense: the marginal cost
of producing one more sample of output once the model is running.

Dividing one forward pass's total by its output length would be wrong, and wrong in
a direction that flatters nothing consistently: inner layers of a dilated stack
compute more time-steps than survive to the output, so the quotient overstates cost
by roughly the ratio of stack depth to receptive field (~10% for A2).

Instead cost is measured as the **slope** of total ops against input length, sampled
at three lengths. For any model whose per-sample work is constant -- every causal
convolutional architecture -- total cost is affine in input length and the slope is
exactly the streaming per-sample cost, with edge effects falling out as the
intercept. Sampling a third point also tests linearity: an architecture whose cost
grows superlinearly (attention over an uncached, growing context) is detected and
flagged rather than silently reported at one arbitrary length.
"""

from __future__ import annotations

import math as _math
from dataclasses import dataclass as _dataclass, field as _field
from typing import Any, Callable, Dict, Sequence, Set

import torch as _torch
from torch.utils._python_dispatch import TorchDispatchMode as _TorchDispatchMode

__all__ = [
    "CostReport",
    "UnregisteredOpError",
    "count_cost",
    "REFERENCE_CONTEXT_SAMPLES",
]

#: Input length (samples) at which cost is measured. Cost numbers are only
#: comparable to each other when measured at the same reference length.
REFERENCE_CONTEXT_SAMPLES = 32768


class UnregisteredOpError(RuntimeError):
    """An op ran that has no cost rule.

    Raised rather than defaulting to zero: an unaccounted op is indistinguishable
    from free compute, and the whole point of the cap is that it cannot be dodged.
    """

    def __init__(self, op_name: str):
        self.op_name = op_name
        super().__init__(
            f"No cost rule for ATen op {op_name!r}. Unknown ops are refused rather "
            f"than counted as free, because a free-by-default rule would let the "
            f"compute cap be bypassed. Add a rule to harness/cost.py if this op is "
            f"legitimate."
        )


@_dataclass
class CostReport:
    """Streaming compute cost, per output audio sample."""

    macs_per_sample: float
    elementwise_per_sample: float
    params: int
    output_samples: int
    total_macs: int
    total_elementwise: int
    op_counts: Dict[str, int] = _field(default_factory=dict)
    #: False when cost is not affine in input length, i.e. per-sample cost depends on
    #: how much context has accumulated. The reported figure is then only valid at
    #: REFERENCE_CONTEXT_SAMPLES and must not be compared against fixed-cost models.
    is_linear: bool = True
    nonlinearity: float = 0.0

    def as_row(self) -> Dict[str, Any]:
        return {
            "macs_per_sample": round(self.macs_per_sample, 2),
            "elementwise_per_sample": round(self.elementwise_per_sample, 2),
            "params": self.params,
        }


def _numel(shape: Sequence[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


# --------------------------------------------------------------------------------
# MAC rules. Each returns multiply-accumulate count for one invocation.
# --------------------------------------------------------------------------------


def _convolution_macs(args, out) -> int:
    # aten.convolution(input, weight, bias, stride, padding, dilation,
    #                  transposed, output_padding, groups)
    weight = args[1]
    groups = int(args[8]) if len(args) > 8 else 1
    in_channels_per_group = int(weight.shape[1])
    kernel_numel = _numel(weight.shape[2:])
    # One MAC per (output element x input channel in group x kernel tap).
    return _numel(out.shape) * in_channels_per_group * kernel_numel


def _mm_macs(args, out) -> int:
    a = args[0]
    return _numel(out.shape) * int(a.shape[-1])


def _addmm_macs(args, out) -> int:
    # aten.addmm(bias, mat1, mat2)
    mat1 = args[1]
    return _numel(out.shape) * int(mat1.shape[-1])


def _bmm_macs(args, out) -> int:
    a = args[0]
    return _numel(out.shape) * int(a.shape[-1])


def _baddbmm_macs(args, out) -> int:
    return _numel(out.shape) * int(args[1].shape[-1])


def _sdpa_macs(args, out) -> int:
    # scaled_dot_product_attention(query, key, value, ...)
    q, k, v = args[0], args[1], args[2]
    d_qk = int(q.shape[-1])
    d_v = int(v.shape[-1])
    lq = int(q.shape[-2])
    lk = int(k.shape[-2])
    batch = _numel(q.shape[:-2])
    # QK^T then (attn @ V)
    return batch * lq * lk * (d_qk + d_v)


_MAC_RULES: Dict[str, Callable[[Any, Any], int]] = {
    "convolution": _convolution_macs,
    "_convolution": _convolution_macs,
    "conv1d": _convolution_macs,
    "conv2d": _convolution_macs,
    "mm": _mm_macs,
    "matmul": _mm_macs,
    "addmm": _addmm_macs,
    "bmm": _bmm_macs,
    "baddbmm": _baddbmm_macs,
    "linear": _addmm_macs,
    "_scaled_dot_product_flash_attention": _sdpa_macs,
    "_scaled_dot_product_efficient_attention": _sdpa_macs,
    "scaled_dot_product_attention": _sdpa_macs,
}

# Pointwise / reduction math: no MACs, but real work. Tracked as a second budget so
# that shifting compute into elementwise gating is visible rather than free.
_ELEMENTWISE: Set[str] = {
    "add", "add_", "sub", "sub_", "rsub", "mul", "mul_", "div", "div_",
    "neg", "abs", "exp", "log", "log1p", "pow", "sqrt", "rsqrt", "reciprocal",
    "tanh", "sigmoid", "relu", "leaky_relu", "gelu", "elu", "silu", "softplus",
    "hardtanh", "clamp", "clamp_min", "clamp_max", "sign", "erf", "sin", "cos",
    "softmax", "_softmax", "log_softmax", "_log_softmax",
    "sum", "mean", "var", "std", "amax", "amin", "max", "min", "prod",
    "native_layer_norm", "native_batch_norm", "native_group_norm",
    "hardsigmoid", "hardswish", "mish", "threshold", "where", "lerp",
    "addcmul", "addcdiv", "fmod", "remainder", "atan", "asin", "acos", "tan",
    "copy_", "fill_", "zero_", "masked_fill", "masked_fill_",
}

# Metadata / memory movement. Genuinely ~free in MAC terms.
_FREE: Set[str] = {
    "view", "_unsafe_view", "reshape", "permute", "transpose", "t", "contiguous",
    "clone", "detach", "alias", "expand", "expand_as", "broadcast_to",
    "unsqueeze", "squeeze", "select", "slice", "narrow", "index", "index_select",
    "cat", "stack", "split", "split_with_sizes", "chunk", "unbind", "unfold",
    "to", "_to_copy", "type_as", "empty", "empty_like", "zeros", "zeros_like",
    "ones", "ones_like", "full", "full_like", "arange", "flip", "roll",
    "pad", "constant_pad_nd", "reflection_pad1d", "replication_pad1d",
    "as_strided", "resize_", "set_", "_reshape_alias", "flatten", "ravel",
    "repeat", "repeat_interleave", "tile", "gather", "scatter", "scatter_",
    "new_empty", "new_zeros", "new_ones", "new_full", "lift_fresh",
    "_local_scalar_dense", "equal", "eq", "ne", "gt", "lt", "ge", "le",
    "isnan", "isinf", "all", "any", "nonzero", "argmax", "argmin", "sort",
    "topk", "cumsum", "roll", "meshgrid", "linspace",
}


class _CostMode(_TorchDispatchMode):
    def __init__(self) -> None:
        super().__init__()
        self.macs = 0
        self.elementwise = 0
        self.op_counts: Dict[str, int] = {}
        self.unknown: Set[str] = set()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)

        name = func.overloadpacket.__name__ if hasattr(func, "overloadpacket") else str(func)
        self.op_counts[name] = self.op_counts.get(name, 0) + 1

        primary = out[0] if isinstance(out, (tuple, list)) and out else out

        if name in _MAC_RULES:
            if isinstance(primary, _torch.Tensor):
                try:
                    self.macs += int(_MAC_RULES[name](args, primary))
                except (IndexError, AttributeError, TypeError):
                    # Shape assumptions failed; refuse rather than undercount.
                    self.unknown.add(name)
        elif name in _ELEMENTWISE:
            if isinstance(primary, _torch.Tensor):
                self.elementwise += _numel(primary.shape)
        elif name in _FREE:
            pass
        else:
            self.unknown.add(name)

        return out


#: Spacing between the input lengths used for the slope fit. Large enough that the
#: op-count difference is dominated by steady-state work rather than rounding.
_PROBE_STRIDE = 4096

#: Relative tolerance on the linearity check.
_LINEARITY_RTOL = 1e-6


def _measure(model, length: int, batch_size: int, device: str, make_inputs) -> tuple:
    """Run one forward pass under the dispatch counter."""
    if make_inputs is None:
        inputs = (_torch.zeros(batch_size, length, device=device),)
    else:
        inputs = tuple(make_inputs(batch_size, length))

    mode = _CostMode()
    with _torch.no_grad():
        with mode:
            out = model(*inputs)

    if mode.unknown:
        raise UnregisteredOpError(", ".join(sorted(mode.unknown)))

    out_tensor = out[0] if isinstance(out, (tuple, list)) else out
    return mode.macs, mode.elementwise, mode.op_counts, int(out_tensor.shape[-1]) * batch_size


def count_cost(
    model: _torch.nn.Module,
    input_samples: int = REFERENCE_CONTEXT_SAMPLES,
    *,
    batch_size: int = 1,
    device: str = "cpu",
    make_inputs: Callable[[int, int], Sequence[_torch.Tensor]] | None = None,
) -> CostReport:
    """Measure streaming per-output-sample compute cost of ``model``.

    Runs three forward passes at increasing input lengths and fits the slope; see
    the module docstring for why the slope rather than a single pass's quotient.

    :param model: The model under test. Called as ``model(x)`` unless
        ``make_inputs`` is supplied.
    :param input_samples: Base probe length. Must exceed the model's receptive field.
    :param make_inputs: Optional factory ``(batch_size, length) -> args`` for models
        whose signature is not ``(x,)``.
    :raises UnregisteredOpError: if any op ran without a cost rule.
    """
    model = model.to(device).eval()

    lengths = [input_samples, input_samples + _PROBE_STRIDE, input_samples + 2 * _PROBE_STRIDE]
    macs, elementwise, counts, out_samples = [], [], None, None
    for L in lengths:
        m, e, c, n = _measure(model, L, batch_size, device, make_inputs)
        macs.append(m)
        elementwise.append(e)
        if counts is None:
            counts, out_samples = c, n

    if out_samples is not None and out_samples <= 0:
        raise ValueError("Model produced no output samples; is the probe shorter than its receptive field?")

    per_sample = lambda ys: [  # noqa: E731
        (ys[i + 1] - ys[i]) / (_PROBE_STRIDE * batch_size) for i in range(len(ys) - 1)
    ]
    mac_slopes = per_sample(macs)
    ew_slopes = per_sample(elementwise)

    # Linearity: equal successive slopes means constant per-sample cost.
    scale = max(abs(mac_slopes[0]), 1.0)
    nonlinearity = abs(mac_slopes[1] - mac_slopes[0]) / scale
    is_linear = nonlinearity <= _LINEARITY_RTOL

    return CostReport(
        macs_per_sample=mac_slopes[-1],
        elementwise_per_sample=ew_slopes[-1],
        params=sum(p.numel() for p in model.parameters()),
        output_samples=out_samples or 0,
        total_macs=macs[0],
        total_elementwise=elementwise[0],
        op_counts=dict(counts or {}),
        is_linear=is_linear,
        nonlinearity=nonlinearity,
    )
