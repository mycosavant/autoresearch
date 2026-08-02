# autoresearch: neural amp modeling

Autonomous architecture search for guitar amp modeling. An agent edits one file,
trains under a fixed budget, is scored against a tamper-proof metric, keeps or
discards, and repeats — overnight, unattended.

The question it exists to answer: **is there a better architecture than A2, at A2's
compute budget?**

Adapted from [karpathy/autoresearch](https://github.com/karpathy/autoresearch),
which applies the same loop to LLM pretraining. Ground truth for the domain is
Steven Atkinson's [NAM](https://github.com/sdatkinson/neural-amp-modeler).

## Why this differs from the upstream loop

Upstream optimizes a single scalar (`val_bpb`) under fixed wall-clock. Amp modeling
is a two-axis problem: A2's achievement is a *Pareto* move — better tone at less
compute, enough to run on a $3 microcontroller — not an accuracy move. Scoring
accuracy alone would reward models that can never run in real time.

So the objective here is **minimize error subject to a hard compute cap**, with the
cap fixed at exactly A2 standard's cost. Every result reads directly as
beats / does-not-beat A2 at equal compute.

Three further changes follow from the domain:

- **The panel.** Every architecture is trained on 4 captures (clean / crunch /
  high-gain / pedal) and scored on the mean, because a model tuned until it excels
  on one amp teaches you nothing about the architecture.
- **The noise floor.** ESR varies run to run for reasons unrelated to architecture.
  Improvements must clear a measured floor, or the loop banks variance for fifty
  iterations while the log shows steady progress.
- **The perceptual guard.** ESR is a known-imperfect proxy for perceived tone. MRSTFT
  is logged every run and rank divergence between the two is tracked, so metric
  gaming shows up instead of quietly accumulating.

## How it works

| File | Role |
|---|---|
| `prepare.py` | Data prep: fetch, align, normalize captures; write the panel manifest. Read-only. |
| `harness/` | Constants, cost accounting, metrics, decision rule, run driver. **Read-only.** |
| `train.py` | Model, optimizer, training loop. **The agent edits this and only this.** |
| `program.md` | Agent instructions. **The human edits this.** |
| `results.tsv` | Run log (untracked). |

### Scoring

Score is panel-mean pre-emphasis ESR (NAM's own definition, `pre_emph_coef=0.85`),
gated on compute:

| | A2 standard | A2 nano |
|---|---|---|
| parameters | 12,145 | 1,870 |
| MACs/sample | **11,776** | 1,731 |
| receptive field | 6,347 samples (132 ms @ 48 kHz) | 6,347 |

A run over budget is `invalid`, not merely worse.

### Why the cost counter is built the way it is

The cap is only meaningful if it cannot be dodged, and the cost axis is analytic and
therefore forgeable. So:

- Ops are counted via `__torch_dispatch__`, not module hooks. A model calling
  `F.conv1d` directly would report zero cost under hooks.
- An op with no cost rule **raises** rather than counting as free. Free-by-default
  is the one failure mode that would silently invalidate every result in the log.
- Elementwise work is capped separately, so compute cannot be moved into giant
  pointwise gating.
- Cost is the *slope* of ops against input length — true streaming per-sample cost.
  A single forward pass's quotient overstates A2 by ~10%, because inner layers of a
  dilated stack compute more time-steps than survive to the output. Sampling a third
  point also detects architectures whose per-sample cost grows with context.
- The cap is recomputed from `harness/reference.py` every run rather than stored, so
  there is no artifact to edit.

## Quick start

**Requirements:** a single NVIDIA GPU, Python 3.10+, [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # install
uv run prepare.py --sources sources.json # fetch + align captures (one-time)
uv run train.py                          # one experiment
```

For the first session on a new box, follow `docs/BASELINES.md` instead — the loop
cannot make a keep/discard decision until an incumbent and a noise floor exist.

Then point an agent at the repo:

```
Have a look at program.md and let's kick off a new experiment.
```

### Note on headless boxes

NAM's package `__init__` imports its Tk-based trainer GUI, so `import nam` needs
`tkinter` present even when nothing is displayed. Install the `python3-tk` matching
your Python version, or `import nam.models.losses` fails on an otherwise fine
training box.

## Baselines first

The loop establishes A2 standard, A2 nano, A1 standard, and the noise floor before
inventing anything. **`docs/BASELINES.md` is the runbook** — the exact command
sequence, about two hours end to end.

There is **no published per-capture A2 ESR figure to check against** — TONE3000
released only MUSHRA listener ratings from its 39-tone evaluation, and the Slimmable
NAM paper's curve describes slimmable A1-family models, not A2 (the two are orthogonal
axes that co-launched). So the gate is internal consistency: A2 must beat A1 at lower
compute, A2 standard must clearly beat A2 nano, cost figures must match the three
independent derivations, and ESR must be in a sane absolute range. See `docs/DATA.md`.
If those fail, the harness is wrong and no result from it means anything.

## A2 provenance

There is no shipped A2 `.nam` file and no A2 preset in the Python trainer —
`example_models/wavenet_a2_max.nam` is a schema stress-test model, not A2, and
`nam/train/core.py`'s `Architecture.NANO` is *A1* nano. `harness/reference.py`
reconstructs A2 from the C++ fast-path detector in
`NeuralAmpModelerCore/NAM/wavenet/a2_fast.{h,cpp}`, which is the only authoritative
definition on disk. The reconstruction is verified three ways: parameter counts
(12,145 / 1,870), the receptive field (6,347), and MACs/sample measured
independently by dispatch counting.

Two traps found while doing this, worth knowing if you touch A2 yourself:

- A2 uses `head_scale = 0.01`. Every A1 preset uses `0.02`. Train A2 with `0.02` and
  the C++ detector silently rejects it and falls back to the generic (much slower)
  WaveNet, with no warning anywhere.
- `generate_weights_a2.py` is not a valid parameter-count oracle for A2: it reads a
  scalar `kernel_size` (A2 uses `kernel_sizes`) and counts the head rechannel as a
  1×1, undercounting it 16-fold.

## Status

M0 — harness complete and tested; capture panel not yet populated. See
`docs/RESEARCH_PLAN.md` for the roadmap, including the Burn (Rust) training backend
on track B behind a gradient-parity gate.

## License

MIT
