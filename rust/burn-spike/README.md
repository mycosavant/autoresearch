# burn-spike

Track B2, step 1: **can Burn express and differentiate the convolution a NAM
WaveNet is made of?**

The research plan named one risk as the thing that sets the estimate — grouped *and*
dilated causal `conv1d` with a working backward pass. If it were missing, B2 would
mean writing a CubeCL kernel rather than porting a trainer.

## Verdict

**Yes. No custom kernel is required for capability reasons.** Burn 0.21.0 supports
`dilation > 1` and `groups > 1` simultaneously, forward and backward, and causal
padding is native.

| Question | Answer | Evidence |
|---|---|---|
| `dilation > 1` and `groups > 1` together, forward | yes | `grouped_dilated_conv1d_matches_the_naive_reference` — checked against an independent implementation, not just "no panic" |
| Groups actually isolate channels | yes | `groups_do_not_leak_across_the_channel_boundary` |
| Backward for that combination | yes | `gradients_exist_for_grouped_dilated_conv` |
| Backward is *correct* | yes | `weight_gradient_matches_central_differences`, f64, rel 1e-6 |
| Causal padding | native | `PaddingConfig1d::Explicit(left, right)` |
| Composes at A2's depth | yes | 23 layers, dilations to 239, mixed kernels 6/15 |
| Trains | yes | Adam, loss decreases |

The plan listed causal padding as something we might have to hand-roll. In 0.21 it is
`Explicit((kernel - 1) * dilation, 0)` — `Conv1d::forward` routes asymmetric padding
through an explicit `pad` op, so output length equals input length and a stack
composes with no per-layer bookkeeping.

## What this does not answer

**Throughput.** The plan's swap criterion is parity *and* at least 0.9x PyTorch. This
spike measures neither speed nor numerical agreement with PyTorch. It establishes
capability only.

That matters more than it might, because the capability answer relocates the risk
rather than removing it. Every grouped convolution on CUDA — forward, dgrad, wgrad —
is declined by the tensor-core implicit-GEMM path and falls back to a naive direct
kernel, with wgrad launching once per group. For depthwise layers (`in_channels /
groups == 1`) vectorization is lost as well. Burn issue
[#4598](https://github.com/tracel-ai/burn/issues/4598) reports a depthwise-heavy net
training ~36x slower than PyTorch on CUDA; root cause unconfirmed, but consistent.
**If grouped layers dominate the architecture, B2 becomes a performance project.**
Measure before committing.

**Anything on a GPU.** Every test here ran on `ndarray`. `cargo check --features
cuda` passes, which is not the same claim.

## Running it

```bash
cargo test                  # CPU, ~80s (the A2-depth test dominates)
cargo test --features cuda  # on a machine with a GPU
```

The `cuda_tests` module re-checks forward, causality and the weight gradient at A2's
real dilations (1, 3, 17, 101, 239) against the `ndarray` backend. Cross-backend
rather than self-consistent, so it can catch a wrong kernel rather than only a
crashing one.

**Do this before writing trainer code.** Burn's own test suite covers `dilation > 1,
groups = 1` and `dilation = 1, groups > 1` — and nothing, anywhere, exercises both at
once, for conv1d or conv2d, forward or backward. The kernels treat the two features
orthogonally and reading them suggests it is fine, but the intersection a NAM stack
sits on is uncovered upstream. It is one command to convert that inference into
evidence.

## Upstream bug found

Grouped conv1d **backward** panics when `stride > 1`:

```
ndarray: could not broadcast array from shape: [2, 2, 4] to: [2, 2, 3]
  at burn-backend-0.21.0/src/backend/ops/modules/conv.rs:951
```

`conv1d_weight_grad_groups` is missing the kernel-size truncation guard that the 2-D
and 3-D versions got in tracel-ai/burn PR
[#3521](https://github.com/tracel-ai/burn/pull/3521) (issue
[#3511](https://github.com/tracel-ai/burn/issues/3511)); the 1-D case appears simply
to have been missed.

Measured trigger:

| groups | stride | padding | backward |
|---|---|---|---|
| 2 | 1 | any of (0,0) (1,1) (2,0) | ok |
| 2 | 2 | any of (0,0) (1,1) (2,0) | **panic** |
| 1 | 2 | any | ok |

`groups > 1` **and** `stride > 1`. Padding symmetry is irrelevant — noted because the
plausible hypothesis (that the asymmetric path dodges it by handing the backend
`padding = [0]`) is wrong, and was written into this crate as fact until the control
test failed.

WaveNet does not stride, so this does not bite us. It is pinned by
`grouped_strided_conv1d_backward_hits_an_upstream_slice_bug`, a `should_panic` test,
so that a future Burn fixing it fails loudly instead of the constraint decaying into
folklore. Worth reporting upstream.

## Also worth knowing

- **`PaddingConfig1d::Same` ignores dilation.** `calculate_same_padding` takes no
  dilation argument, so with `dilation > 1` it will not preserve length. Use
  `Explicit`. We want `Explicit` for causality anyway.
- **The `ndarray` backend is slow at this shape.** One forward pass of an A2-depth
  stack over 8,192 samples takes over a minute. Fine for correctness, useless for
  timing — which is the other reason throughput has to be measured on the GPU.

## Next

1. Run `cargo test --features cuda` on the research box. Cheap, and it converts the
   central capability claim from code inspection to measurement.
2. Benchmark grouped vs dense layers on that hardware — the actual open risk.
3. Only then the parity harness: weights imported from a shared `.npz`, identical
   batch order, forward and gradient agreement against PyTorch, then a 200-step loss
   curve.
