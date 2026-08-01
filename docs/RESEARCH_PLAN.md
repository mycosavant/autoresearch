# Autoresearch for Neural Amp Modeling

## Context

The goal is an autonomous research loop that attacks neural amp modeling architecture the way
Karpathy's `autoresearch` attacks LLM pretraining: the agent edits one file, trains under a fixed
budget, scores against a tamper-proof metric, keeps or discards, and repeats overnight.

Ground truth is Steven Atkinson's NAM (note: *Atkinson*, not "Steve Atkins"). Three repos are
already cloned and in scope:

- `mycosavant/autoresearch` — fork of Karpathy's LLM-pretraining loop. Scaffold to be retargeted.
- `mycosavant/neural-amp-modeler` — the Python trainer. Already supports per-layer kernel sizes,
  FiLM, gating modes, layer-array heads, and slimmable widths, so a wide search space is expressible today.
- `mycosavant/NeuralAmpModelerCore` — C++ inference. Contains the A2 fast path and `benchmodel`.

**The A2 shape is known exactly** from `NAM/wavenet/a2_fast.h`: 23 layers, channels 3 (nano) / 8
(standard), kernel size 6 with two 15s at layers 14–15, dilations `1,3,7,17,41,101,239` repeated
(with a `1,13` interlude), head rechannel kernel 16, head scale 0.01, LeakyReLU 0.01. Receptive
field 6,347 samples (~132 ms @ 48 kHz). Verified against a real deployed A2 model
(`BossWN-a2.nam`): 12,145/1,870 params, 11,776/1,731 MACs/sample.

**The central design departure from Karpathy's loop:** his metric is one scalar (`val_bpb`) under
fixed wall-clock. Amp modeling is two-axis — A2's achievement is a *Pareto* move, not an accuracy
move. So the objective here is **minimize error subject to a hard compute cap**, with the cap set to
exactly A2-standard's MACs/sample. Every result then reads directly as "beats / does not beat A2 at
equal compute."

Decisions: local NVIDIA GPU; composite metric with a perceptual guard; Slimmable-NAM paper captures
plus ToneTwist research datasets (TONE3000 turned out to publish only trained models, never training
audio); Rust work sequenced inference-first, PyTorch experiments starting immediately.

Cost axis revised after evidence: originally MACs-primary, now a **calibrated runtime model**.
modulus measured a 6.8x MAC increase producing only 2.17x more runtime at A2's channel widths, so
raw MACs systematically undervalue width and overvalue depth. See `docs/CALIBRATION.md`.

## Approach

Retarget the `autoresearch` fork in place, on branch `claude/neural-modeling-architecture-vkz3v8`.
Keep Karpathy's loop discipline (read-only harness / one agent-editable file / `results.tsv` /
keep-discard / branch advance) and replace the LLM internals.

### Repo layout (`mycosavant/autoresearch`)

| File | Role |
|---|---|
| `prepare.py` | **Read-only, tamper-proof.** Data prep, fixed constants, `evaluate()`, and the MAC counter. |
| `train.py` | **The only file the agent edits.** Model, optimizer, training loop. |
| `cost.py` | Merged into `prepare.py` deliberately — cost accounting must not be agent-editable. |
| `program.md` | Agent instructions, rewritten for this domain. |
| `results.tsv` | Untracked run log, extended schema. |
| `harness/cost_model.py` | Calibrated runtime model; refuses to gate while unidentified. |

The existing LLM `train.py`/`prepare.py` are not reusable and get replaced. Keep the README's credit
to the upstream project.

### Objective and scoring

Score = **mean pre-emphasis ESR across a fixed panel of captures**, gated on
`macs_per_sample <= BUDGET`. Over budget is `invalid`, not merely worse — otherwise the agent buys
accuracy with compute.

`evaluate()` in `prepare.py` returns a bundle, reusing NAM's existing implementations
(`nam/models/losses.py`: `esr`, `apply_pre_emphasis_filter`, `multi_resolution_stft_loss`):

- `esr` — headline, pre-emphasis filtered, mean over panel
- `mrstft` — multi-resolution STFT, logged every run
- `esr_holdout` — amps never trained on, generalization guard
- `macs_per_sample`, `params`, `receptive_field_ms`

**Panel, not single capture.** A fixed panel of 4 captures spanning clean / crunch / high-gain /
pedal, each trained fresh per experiment, mean ESR as the score. Optimizing a single capture would
overfit the *architecture* to one amp's nonlinearity — the failure mode that makes an
architecture search worthless.

**Budget:** fixed wall-clock per experiment (default 10 min, constant in `prepare.py`), split across
the panel. ~6 experiments/hour, ~50 overnight.

### Anti-gaming guardrails

These matter more here than in the LLM loop, because the cost axis is analytic and therefore forgeable.

1. MAC cap enforced in read-only code, as a hard gate.
2. MAC counter traces the agent's model against an op-cost registry. **Unknown op ⇒ run marked
   `invalid`**, so compute cannot be hidden in an unaccounted operation.
3. `esr_holdout` reported every run; headline-improves-but-holdout-degrades is flagged.
4. **Noise floor.** ESR run-to-run variance is real and Karpathy's loop has no defense against it.
   Measure it once by running the baseline 3× with different seeds; require improvements to exceed
   it before `keep`. Without this the agent banks noise for 50 iterations and drifts.
5. Perceptual guard: track rank correlation between the ESR ordering and the MRSTFT ordering across
   the run log. Sustained divergence means ESR is being gamed and gets surfaced, not silently kept.
6. **Streaming-form verification** (`harness/streaming.py`). Recurrent architectures have to be
   costed on something other than what they trained through, which is a hole in the cap by
   construction. A `streaming_form()` is believed only after it matches the training form to better
   than 1e-6 ESR, adds no parameters, and survives an extrapolation of its short-probe cost fit
   against a real measurement at the reference length. Failing any check raises rather than falling
   back — a fallback would log "too expensive" where the truth is "unverifiable".

### Baselines before novelty

First runs reproduce, not invent: A2 standard (the number to beat), A2 nano, A1 standard, LSTM.
Then the noise-floor measurement. Only then does the search open up.

### Track B — Rust, inference first

Re-scoped after surveying the `modulus` repo, which turns out to be a mature
(~130k LOC, 15 crates) real-time Rust neural-amp platform with NAM A2 inference
already passing parity against the C++ reference at ~4e-7, CI-gated.

**B1 — Novel architectures as `ModelBackend` impls (first).** modulus's
`ModelBackend` trait (`crates/inference-traits/src/lib.rs`) is architecture-agnostic:
five methods, nothing NAM-shaped. A state-space or IIR backend implements it without
touching the format parser, and candidates can carry our own format rather than going
through `model-formats`. This is what unblocks the cost axis — every backend gives a
real measured runtime point, which is what the calibration needs. Follow
`.claude/skills/add-inference-backend.md`; reuse `tests/rt_safety_inference.rs` as a
ready-made RT-safety gate.

**B2 — Burn trainer (after).** modulus has *zero* training code — no burn, candle or
tch, inference-only by ADR-0009 — so this remains greenfield either way. Sequenced
behind B1 because B1 has the clearer payoff. Unchanged in substance:

1. **Spike first:** confirm Burn's `conv1d` supports dilation *and* groups with
   working CUDA autodiff, plus causal padding. Grouped dilated conv is the risk; if
   it is missing, a CubeCL kernel is required and that changes the estimate.
2. **Parity harness:** fixed seed, weights imported from a shared `.npz`, identical
   batch order. Compare forward outputs and gradients against PyTorch, then a
   200-step loss curve.
3. **Swap criterion:** parity holds AND throughput >= 0.9x PyTorch. Published
   Burn-vs-PyTorch benchmark numbers are unverified blog claims -- measure locally.

Two constraints inherited from modulus worth knowing before either step. Its
bit-parity-against-C++ policy forecloses FMA, reduction reordering and therefore
quantization; a quantized result from this search cannot be validated by those gates
and needs its own tolerance regime. And there is no ARM target anywhere in the tree,
so any embedded claim is unmeasured.

### RTF audit

Feeds the calibration rather than standing alone. Benchmark candidates via modulus
(`cargo run -p test-harness --release --bin nam_diag`, note the underscore) and add the points to
`MODULUS_MEASUREMENTS`. Once the design matrix reaches full rank with >=5 points, the
predicted-runtime cap begins binding automatically.

**Known limitation:** architectures with no Rust backend cannot be measured, so the calibration only
thickens as fast as B1 delivers backends. Until then the MAC and elementwise caps are the only hard
constraints, with predicted runtime reported but advisory.

## Milestones

- **M0** — Scaffold, data prep, panel + holdout split, MAC counter, `evaluate()`. Baselines
  reproduced, noise floor measured.
- **M1** — PyTorch loop live, `program.md` rewritten, first autonomous night.
- **M2** — Burn spike + parity harness.
- **M3** — Backend swap behind the parity gate.
- **M4** — RTF audit wired into the loop.

## Verification

- `uv run prepare.py` completes; shards + panel manifest land in the cache dir.
- `uv run train.py` finishes inside the budget and prints the metric bundle.
- **A2 reproduction check:** harness-trained A2-standard lands within the measured noise floor of
  the published A2 ESR on the same captures. If it does not, the harness is wrong and no result from
  it means anything — this gates everything downstream.
- MAC counter cross-checked by hand against the known A2 shape from `a2_fast.h`.
- Guardrail tests: an over-budget model marks `invalid`; a model using an unregistered op marks
  `invalid`.
- Parity harness: `cargo test` gradient comparison passes before any backend swap.

## Note on this container

No GPU, no torch, 4 CPUs. This box builds and unit-tests the harness; the loop runs on your CUDA
machine. Data prep and baseline runs happen there.
