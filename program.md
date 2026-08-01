# autoresearch: neural amp modeling

Autonomous architecture search for guitar amp modeling. You are the researcher.

The question this repo exists to answer: **is there a better architecture than A2, at
A2's compute budget?** Everything else is in service of that.

## Setup

1. **Agree a run tag** with the human, e.g. `aug1`. The branch `autoresearch/<tag>`
   must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>`.
3. **Read the in-scope files**:
   - `README.md` — repo context.
   - `harness/` — the read-only harness. Read it, so you know exactly how you are
     being scored. Do not modify it.
   - `train.py` — the file you modify.
4. **Verify data**: check `~/.cache/nam-autoresearch/` has capture shards and
   `panel.json`. If not, tell the human to run `uv run prepare.py`.
5. **Initialize `results.tsv`** with the header row only.
6. **Confirm** setup looks good, then begin.

## What you may and may not change

**You edit exactly one file: `train.py`.** Inside it everything is fair game —
architecture, optimizer, schedule, batch size, augmentation, initialization, and the
loss used *for training*.

**You may not touch:**

- `harness/` — constants, cost accounting, metrics, the keep/discard rule.
- `prepare.py` — data preparation.
- The evaluation metric. `harness/metrics.py` is ground truth.
- Dependencies. Use what is already in `pyproject.toml`.

If you find yourself wanting to change the harness to make a result look better,
that is the signal that you are about to invalidate the entire run. Don't. Put the
frustration in the description field and move on.

## How you are scored

Score is **panel-mean pre-emphasis ESR, lower is better**, subject to a **hard
compute cap**.

- The cap is A2-standard's cost: **11,776 MACs/sample** (12,145 params, receptive
  field 6,347 samples ≈ 132 ms @ 48 kHz). It is recomputed from
  `harness/reference.py` on every run rather than read from a file, so it cannot
  drift or be edited.
- Exceed it and the run is **`invalid`** — not "worse", *invalid*. An over-budget
  model answers a different question than the one being asked.
- There is also an elementwise-op cap (A2 standard uses 729 pointwise ops/sample), so
  moving work into large elementwise operations does not dodge the MAC cap.
- Cost is measured as the **slope** of ops against input length, i.e. true streaming
  per-sample cost. If your architecture's cost is superlinear in context length
  (attention over an uncached growing context), it is flagged `is_linear=False` and
  cannot be fairly compared — you will need to give it a bounded-cost streaming form.

**Panel, not single capture.** Every model is trained on 4 captures (clean / crunch /
high-gain / pedal) and scored on the mean. A win on one amp that loses on the others
is not a win. Scoring a single capture would overfit the *architecture* to one amp's
nonlinearity, which is the failure mode that makes an architecture search worthless.

**`esr_holdout` is reported, not optimized.** It is measured on amps never trained
on. Do not tune against it — that would destroy the only unbiased generalization
signal in the whole setup.

**Improvements must clear the noise floor.** ESR moves run to run for reasons that
have nothing to do with architecture (init, data order, nondeterministic kernels).
The floor is measured once by running the baseline 3× with different seeds. An
"improvement" smaller than that is not evidence, and is discarded. Do not argue with
this — banking noise is exactly how a loop drifts for 50 iterations while the log
shows steady progress.

## First runs: baselines, not ideas

Invent nothing until the baselines are logged. Set `BASELINE` in `train.py` and run
each. All four are wired and their costs are
verified:

| baseline | MACs/sample | params | receptive field |
|---|---|---|---|
| `a2_standard` | 11,776 | 12,145 | 6,347 |
| `a2_nano` | 1,731 | 1,870 | 6,347 |
| `a1_standard` | 13,320 | 13,801 | 4,093 |
| `lstm` | 51 | 82 | 1 |

Then the **noise floor**: A2 standard, 3 seeds.

Note what A1 vs A2 already tells you — A1 costs *more* (13,320 vs 11,776) for a
receptive field barely two thirds as long (4,093 vs 6,347). That gap is the size of
the move A2 made, and roughly the size of the move you are being asked to find again.

**There is no published per-capture A2 ESR to check against.** TONE3000's 39-tone
evaluation released only MUSHRA listener ratings, and the Slimmable NAM paper's curve
describes slimmable A1-family models, not A2. So the gate is internal consistency:

1. A2 standard must beat A1 standard, at lower compute.
2. A2 standard must clearly beat A2 nano.
3. Costs must match the table above.
4. ESR must be in a sane absolute range for a converged amp model.

If any of those fail, **stop and tell the human**. The harness is wrong, and no
result out of it means anything until that is fixed.

## Output format

`uv run train.py > run.log 2>&1`, then:

```
grep "^esr:\|^macs_per_sample:\|^status:" run.log
```

The script prints:

```
---
esr:              0.004213
mrstft:           0.118400
esr_holdout:      0.005901
macs_per_sample:  11776.0
elementwise:      729.0
params:           12145
rtf:              n/a
training_seconds: 600.2
status:           ok
```

## Logging results

Append to `results.tsv` (tab-separated — commas break descriptions). Leave it
untracked by git.

```
commit	esr	mrstft	esr_holdout	macs_per_sample	params	rtf	status	description
```

`status` ∈ {`keep`, `discard`, `crash`, `invalid`}. Use `0.000000` for metrics that
do not exist because the run failed.

## The experiment loop

LOOP FOREVER:

1. Check git state — current branch and commit.
2. Edit `train.py` with one idea. **One change at a time**; confounded experiments
   teach you nothing.
3. `git commit`.
4. `uv run train.py > run.log 2>&1` — redirect, never `tee`, or the output floods
   your context.
5. `grep "^esr:\|^macs_per_sample:\|^status:" run.log`.
6. Empty grep means a crash. `tail -n 50 run.log` for the traceback. Fix it if it is
   something dumb (typo, missing import). If the idea is fundamentally broken, log
   `crash` and move on.
7. Record in `results.tsv`.
8. `keep` → advance the branch. `discard` / `invalid` → `git reset` back.

**Timeout**: a run should take ~10 minutes plus overhead. Kill anything past 30
minutes and treat it as a failure.

**NEVER STOP**: once the loop has begun, do not pause to ask whether to continue. The
human may be asleep. If you run out of ideas, think harder — re-read the harness,
re-read the references below, combine previous near-misses, try something
structurally radical. The loop runs until you are interrupted.

## Research directions

Seeds, not a menu. At a fixed MAC budget you win by *needing* less compute, not by
spending it better — so the interesting moves are structural.

### The big structural bet

A guitar amp is, to first order, a Wiener–Hammerstein system: linear filter → static
nonlinearity → linear filter, plus slow state (power-supply sag, bias drift, speaker
thermal effects). A dilated conv stack spends an enormous number of MACs brute-forcing
long *linear* memory that a second-order IIR section reproduces in ~5 MACs/sample.
A2 spends 11,776 MACs/sample to reach 132 ms of context. That ratio is the
opportunity.

- **Learned IIR / biquad cascades** for the linear blocks, keeping dilated convs only
  for what is genuinely nonlinear. Backprop through a recursive filter is the hard
  part — look at frequency-sampling design and differentiable-DSP work.
- **Linear state-space layers** (S4 / Mamba-style diagonal SSMs). A linear recurrence
  gives unbounded receptive field at O(1) per sample. Arguably the most natural fit
  for amp modeling and badly under-explored here. Note the cost counter measures
  streaming slope, so a properly-formulated recurrence is scored fairly.
- **Explicit Wiener–Hammerstein blocks** as an architectural prior, rather than
  making a generic stack rediscover the structure from data.

### Cheaper convolution

Depthwise-separable and grouped convs. A2 is already narrow (8 channels) so grouping
may not pay — but measure it rather than assuming. Note that grouped convs and any
`bottleneck != channels` drop the model off the C++ fast path, which matters for the
RTF audit but not for scoring.

### Dilation schedule

A2 uses `1,3,7,17,41,101,239` (≈2.4× growth) rather than powers of two, repeated
three times, with a `k=15, d=1` / `k=15, d=13` interlude at layers 14–15. Is any of
that optimal? The series is "gapless" for k=6 — each dilation never exceeds the
cumulative receptive field before it. Test growth rates, and test whether the two
kernel-15 layers are load-bearing.

### Activation

A2 uses LeakyReLU(0.01); A1 used Tanh. Gated activations double conv cost — do they
earn it at fixed MACs? Consider cheap polynomial or piecewise nonlinearities, which
are closer to what the analog circuit actually does.

### Receptive field

A2's is 6,347 samples. Is that necessary? Measure the ESR cost of halving it and
spending the freed MACs on width or depth.

### Training-side wins

Free at inference, so they can never breach the cap: optimizer, LR schedule, loss
weighting, augmentation, longer effective context, initialization, weight EMA.

### Quantization-aware training

A model trained int8-aware may beat a float model quantized after the fact, and low
precision is one of the few levers that reduces real cost rather than just MAC count.

Two caveats worth knowing before spending a night on this. First, the "A2 runs on a
$3 Cortex-M7" figure is **unverified vendor marketing** — it appears once in the
modulus research trail, flagged as corroboration to be re-measured, and never was. No
ARM measurement exists in either codebase. Do not treat it as a target you are
hitting or missing. Second, quantization breaks bit-parity against the C++ reference,
which is a hard product constraint for an inference runtime but *not* a constraint on
this search — just be aware that a quantized win here cannot be validated by the
existing parity gates and needs its own tolerance regime.

## Reading

- A2 architecture: `harness/reference.py`, and
  `NeuralAmpModelerCore/NAM/wavenet/a2_fast.{h,cpp}` for the authoritative detector.
- Slimmable NAM: arXiv 2511.07470.
- Wright et al. (Aalto), real-time RNN amp emulation — origin of the ESR metric.
- Pre-emphasis ESR: mdpi.com/2076-3417/10/3/766.
