# Capture data

Provenance, licensing and gotchas for the capture panel. Read this before publishing
any number produced by this harness.

## TONE3000 cannot be used as a training source

Worth stating first, because it is the obvious place to look and it does not work:
**TONE3000 distributes trained `.nam` model files, not the dry/wet audio pairs they
were trained from.** Its API (`/api/v1/tones/*`) exposes models, IRs and alternate
model formats; no endpoint returns training audio. Its terms additionally prohibit
automated bulk downloading and redistribution, and the bulk endpoint is gated to
approved partners.

It remains useful as a source of reference *trained models* — for example
sdatkinson's own Fender Deluxe Reverb A2 capture — but not for training.

## The panel

Slots 1–3 are the **exact captures from the Slimmable NAM paper** (arXiv 2511.07470),
published via its companion repo
[SlimmableNamTrain](https://github.com/Atkinson-Advanced-Modeling/SlimmableNamTrain)
(MIT). This is what makes the A2 reproduction gate meaningful: the same audio, the
same splits, and a published ESR curve to land on.

| Slot | Capture | Device | Source |
|---|---|---|---|
| clean | `clean.wav` | Fender Deluxe Reverb | A2 paper Drive folder |
| crunch | `crunch.wav` | Morgan MVP23 | A2 paper Drive folder |
| high-gain | `rhythm.wav` | Omega Ampworks Obsidian | A2 paper Drive folder |
| pedal | `V100_T050_O050_B000` | Fulltone Full Drive 2 | ToneTwist, Zenodo 10794615 |

The A2 set contains no pedal, hence slot 4 from ToneTwist. `lead.wav` (Omega Obsidian
lead) is carried in the holdout so that all four published points are reproducible.

All A2 files are 48 kHz / 24-bit / mono / exactly 207.000 s.

**Official splits** (from the paper's notes document): train 10–181 s, validation
181–190 s, **test 191–207 s**. The published configs leave the 16-second test region
unused — it is available for clean benchmarking.

## Holdout

Deliberately different amps, dry signals, guitars, and sample rates, so that
"generalizes" means something. All ToneTwist records, CC-BY-NC-4.0:

- **Blackstar HT1 (overdrive)** — Zenodo 10794425. The Wright et al. DAFx-19
  reference set, so RNN-vs-NAM literature comparison is possible.
- **Mesa Mark V (extreme)** — Zenodo 10796864.
- **EHX Big Muff** — Zenodo 10891515. Fuzz: topologically unlike anything in the
  panel, and a deliberately hard case.
- **Omega Obsidian (lead)** — A2 paper set.

## Licensing — read this

- **A2 paper captures: no stated license.** The *code* is MIT; the Drive folder
  holding the audio has no license file. It is deliberately published for
  reproduction, but formal reuse terms are unstated. Email Steven Atkinson before
  publishing derived numbers commercially.
- **ToneTwist: CC-BY-NC-4.0.** Note the repo's MIT license covers the *repository*,
  not the data. **NonCommercial** — fine for research, a problem for commercial use.
- **NAM's demo `output.wav`: no stated license.** Usable as a smoke test; keep it out
  of published results.
- **IDMT-SMT-Audio-Effects is CC BY-NC-ND** — the NoDerivatives clause makes it a
  genuine legal hazard for a training harness. Deliberately not used here.

## Gotchas

**The A2 `input.wav` is not a NAM standardized input.** It is 207 s and matches no
known NAM input version hash (v3.0.0 is 190 s). It works fine with explicit data
configs — which is what the paper does — but NAM's `nam` CLI and GUI auto-detection
will reject it. This harness sidesteps the issue entirely by loading wavs directly
rather than going through NAM's version detection.

**Sample rates differ across the holdout.** ToneTwist "external" devices are
44.1 kHz / 16-bit (they carry their own dry signal at the original source's rate),
while the panel is 48 kHz / 24-bit. `prepare.py` resamples everything to 48 kHz,
NAM's operating rate. Be aware that resampling the holdout is itself a small domain
shift.

**ToneTwist "internal" devices** (the pedals) share a common dry set that *is* the
NAM v2.0.0 standardized input, 48 kHz / 32-bit float / 191.000 s. Those drop into the
NAM trainer with no custom configuration.

**Download size** is roughly 2 GB for the recommended panel and holdout, dominated by
the ToneTwist dry set (866 MB) and Full Drive 2 (1.6 GB, containing many control
settings of which we need one). `prepare.py` caches archives so this is paid once.

## What is not available

- The **39-tone A2 objective evaluation set** has not been released as audio. The
  March 2026 announcement promised evaluation datasets; so far only the MUSHRA
  listener-ratings CSV ([a2-mushra-data](https://github.com/tone-3000/a2-mushra-data),
  CC BY 4.0 — 105,842 ratings, no audio, no per-tone ESR table) and the test harness
  code are public. The four Slimmable-NAM captures are the only A2-paper audio
  obtainable today.
- NAM standardized inputs **v1.0.0 and v1.1.1** have no public URL; the trainer's own
  URL map has them blank.
