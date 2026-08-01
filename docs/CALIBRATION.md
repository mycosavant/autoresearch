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

## Calibration is per-machine

Coefficients encode a specific CPU's arithmetic throughput and per-call overhead.
They do **not** transfer. Calibrations therefore live as named profiles under
`harness/calibration/`, selected by `CALIBRATION_PROFILE` in `harness/constants.py`.

**The repo ships with no profile selected.** Predictions are then unavailable and the
runtime cap does not gate; only the MAC and elementwise caps bind. That is deliberate
— silently applying another machine's coefficients produces confident wrong numbers,
which is worse than no model.

Run `tools/calibrate.py` on the research box (and on any deployment target you care
about) to produce its profile.

## Worked example

`tools/calibrate.py --profile devcontainer-4cpu` generated eight WaveNet variants
spanning depth and width, benchmarked each with modulus's `nam_diag`, and fitted:

| variant | layers | ch | MACs | convs | µs/block |
|---|---|---|---|---|---|
| shallow_narrow | 8 | 8 | 3,784 | 26 | 121 |
| shallow_wide | 12 | 16 | 21,968 | 38 | 530 |
| a2_lite | 23 | 3 | 1,569 | 71 | 110 |
| mid | 23 | 5 | 4,225 | 71 | 192 |
| a2_full | 23 | 8 | 10,624 | 71 | 326 |
| deep_wide | 34 | 6 | 8,874 | 104 | 378 |
| very_deep_narrow | 46 | 2 | 1,414 | 140 | 172 |
| deep_narrow | 46 | 3 | 3,087 | 140 | 230 |

The decisive pair is `deep_narrow` vs `shallow_narrow`: **3,087 MACs takes 230 µs
while 3,784 MACs takes 121 µs.** More arithmetic, nearly half the time. A MAC-only
model gets that backwards.

Fitted result (all four coefficients identified, cond 3.2e4):

| model | worst relative error |
|---|---|
| MAC-only | **75.9%** |
| calibrated | **5–8%** |

5–8% sits inside the measurement spread (2–14% on this shared CPU). The fitted
per-convolution overhead is ~1.0e-2 µs — equivalent to roughly **35 MACs**. A2 Lite's
dilated conv is only 54 MACs per layer, so it sits barely above its own call
overhead. That is the mechanism behind "overhead-bound, not throughput-bound".

### Known imperfection

The fitted intercept is **negative** (≈ −0.2 µs), which is physically meaningless — a
model cannot have negative fixed cost. It indicates mild mis-specification: `conv_ops`
and `elementwise` are correlated across these variants (both grow with depth), so the
fit trades them off against the intercept. The predictions remain accurate in the
region covered by the measurements, but the coefficients should not be read
individually as physical quantities, and extrapolating far outside the measured
range is not safe. Adding variants that break the conv/elementwise correlation would
fix it.

## The degenerate profile, for contrast

`modulus-i9-14900f` carries the two published A2 points. They are **structurally
identical** — both 23 layers and 71 convolutions, differing only in width. So
`conv_ops` never varies and its coefficient is indistinguishable from the intercept:

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
degrees of freedom. The profile is retained for provenance and reports itself
inadequate.

## Calibrating a new machine

```sh
# 1. build modulus's benchmark tool (once)
cd /workspace/modulus && cargo build -p test-harness --release --bin nam_diag

# 2. generate variants, benchmark, fit, write the profile
cd /path/to/autoresearch
python tools/calibrate.py --profile my-workstation --machine "RTX 4090 box, i7-13700K"

# 3. select it
#    CALIBRATION_PROFILE = "my-workstation"   in harness/constants.py
```

The tool reports identifiability and refuses to claim adequacy if the variants do not
separate the coefficients. It writes the profile either way, but an inadequate one
stays advisory.

No Rust backend is needed for this: the variants are ordinary WaveNets exported to
`.nam`, which modulus already loads. Novel architectures (state-space, IIR) *do* need
a `ModelBackend` implementation before they can be measured — see
`.claude/skills/add-inference-backend.md`. Its trait
(`crates/inference-traits/src/lib.rs`) is architecture-agnostic, five methods, nothing
NAM-shaped.

Weights are left at initialization. That is sound for timing because none of these
kernels branch on weight values, so runtime is a function of shape alone. It would
**not** be sound for anything with data-dependent control flow — early exit, dynamic
sparsity, or conditional computation.

Note the binary is **`nam_diag`** (underscore). The docs in
`crates/test-harness/CLAUDE.md` and the tool's own usage header both say `nam-diag`,
which does not resolve — there is no `[[bin]]` stanza, so cargo autodiscovers the
underscore stem.

`nam_diag` is explicitly labelled ad-hoc and is not a gate. For calibration it needs
percentiles rather than mean-and-max, core pinning, and warm-up discard. `criterion`
is already a declared workspace dependency with **zero** users and no `benches/`
directory, so wiring a first benchmark there would be uncontroversial.

A profile flips to `is_adequate` on its own once the design matrix has full rank,
at least five points, and acceptable conditioning — at which point the
predicted-runtime cap starts binding automatically.

## Caveats worth keeping in view

- **One block size.** All numbers are block 64, single-threaded, median-of-runs. The
  `devcontainer-4cpu` profile is a shared virtualized CPU and is a worked example of
  the procedure, not a deployment target.
- **x86 only.** There is no ARM measurement anywhere in either codebase. The "A2 runs
  on a $3 Cortex-M7" figure is vendor marketing, recorded once in modulus's research
  trail flagged for re-measurement, and never re-measured. Do not calibrate against
  it or treat it as a target.
- **The model is linear.** Cache-cliff effects — where a model stops fitting in L1/L2
  — are not linear in any of these features. If predictions degrade sharply for large
  architectures, that is the likely cause and the model needs a capacity term.
