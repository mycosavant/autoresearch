# Calibrating the cost model

## Why MACs alone are not enough

Measured on real A2 models (modulus, i9-14900F, block 64):

| | MACs/sample | µs/block | ×RT |
|---|---|---|---|
| A2 Full (8ch) | 11,776 | 202 | 6.6 |
| A2 Lite (3ch) | 1,731 | 55 | 24.2 |

**6.8× the MACs, 2.17× the time.** In C++ it is worse — 1.61×. The dilated
convolution is ~85% of the arithmetic, but its inner reduction is only 3–8 wide, so
the runtime is dominated by fixed per-operation overhead rather than throughput.

A search scored on MACs alone therefore carries a known, directional bias: it
undervalues width and overvalues depth, because per-layer overhead is invisible to
it. An architecture that trades channels for layers looks free and is not.

## What the model does

`harness/cost_model.py` fits microseconds-per-sample as a linear function of
structural features — `macs`, `elementwise`, `conv_ops`, and a constant — extracted
automatically from the dispatch counter.

It fits by least squares **with an explicit rank and conditioning check**, and reports
any coefficient the measurements cannot separate as *unidentifiable* rather than
returning a number that merely minimises residuals.

## Current state: not adequate to gate on

The two shipped calibration points are **structurally identical** — A2 Full and A2
Lite are both 23 layers and 71 convolutions, differing only in width. So `conv_ops`
never varies, and its coefficient is indistinguishable from the intercept:

```
cost model (2 measurements, cond=68.1, rel. residual=0.000%)
  adequate to gate on: False
  macs         +0.000133773   [UNIDENTIFIABLE]
  elementwise  +0.00209478    [UNIDENTIFIABLE]
  conv_ops     +0.000758207   [UNIDENTIFIABLE]
  const        +1.0679e-05    [UNIDENTIFIABLE]
  note: 'conv_ops' is constant across all measurements …
  note: design matrix is rank-deficient (2 < 4) …
```

The zero residual is *interpolation*, not evidence — two points, two effective
degrees of freedom.

**Consequence:** the predicted-runtime figure is printed on every run but does
**not** gate. Only the MAC and elementwise caps bind. Gating on an unidentified model
would be no more correct than gating on raw MACs and considerably harder to see
through when a result looks wrong.

## How to make it adequate

Needed: **≥5 measurements that vary depth independently of width.** Concretely,
benchmark architectures whose `conv_ops` differs — A1 standard (different layer
count), A2 at reduced depth, and a couple of deliberately deep-and-narrow variants.

Each measurement needs a Rust `ModelBackend` implementation in modulus. Its
`ModelBackend` trait (`crates/inference-traits/src/lib.rs`) is architecture-agnostic —
five methods, nothing NAM-shaped — so a novel backend implements it without touching
the format parser. Follow `.claude/skills/add-inference-backend.md`.

Measure with:

```sh
cargo run -p test-harness --release --bin nam_diag -- \
    --model <model>.nam --input <input>.wav \
    --block-size 64 --sample-rate 48000 --repeats 4
```

Note the binary is **`nam_diag`** (underscore). The docs in
`crates/test-harness/CLAUDE.md` and the tool's own usage header both say `nam-diag`,
which does not resolve — there is no `[[bin]]` stanza, so cargo autodiscovers the
underscore stem.

`nam_diag` is explicitly labelled ad-hoc and is not a gate. For calibration it needs
percentiles rather than mean-and-max, core pinning, and warm-up discard. `criterion`
is already a declared workspace dependency with **zero** users and no `benches/`
directory, so wiring a first benchmark there would be uncontroversial.

Then add the points to `MODULUS_MEASUREMENTS` and re-run the tests. The model flips
to `is_adequate` on its own once the design matrix has full rank, at least five
points, and acceptable conditioning — at which point the predicted-runtime cap starts
binding automatically.

## Caveats worth keeping in view

- **One machine, one block size.** All current numbers are i9-14900F at block 64,
  single-threaded, mean-of-run. No confidence intervals.
- **x86 only.** There is no ARM measurement anywhere in either codebase. The "A2 runs
  on a $3 Cortex-M7" figure is vendor marketing, recorded once in modulus's research
  trail flagged for re-measurement, and never re-measured. Do not calibrate against
  it or treat it as a target.
- **The model is linear.** Cache-cliff effects — where a model stops fitting in L1/L2
  — are not linear in any of these features. If predictions degrade sharply for large
  architectures, that is the likely cause and the model needs a capacity term.
