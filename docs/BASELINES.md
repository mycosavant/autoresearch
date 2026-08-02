# Running the baselines

Everything this harness produces is gated on this sequence. Until it runs, `wh_ssm`'s
10,160 MACs/sample is an architecture fact, not a result — there is no incumbent to
beat and no noise floor to beat it by.

Budget roughly **2 hours of wall clock** and **~2 GB of download**, mostly unattended.

---

## 0. Prerequisites

An NVIDIA GPU with a working driver, `uv`, and ~5 GB free (2 GB of captures, the rest
CUDA wheels). The torch build is pinned to cu128 in `pyproject.toml`, so `uv sync`
fetches the right wheel without a manual index flag.

```bash
git clone https://github.com/mycosavant/autoresearch
cd autoresearch
git checkout claude/neural-modeling-architecture-vkz3v8
uv sync
```

If the box is headless, install `tkinter` first — NAM's package `__init__` imports its
Tk-based trainer GUI, so `import nam` fails without it even though nothing is
displayed, and every command below goes through `nam.models.losses`:

```bash
sudo apt install python3-tk      # match your Python version
```

Confirm the GPU is actually visible before spending an hour finding out it wasn't:

```bash
uv run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

`train.py` falls back to CPU silently if this is `False` — it will run, and it will be
useless, because the time budget is wall-clock.

---

## 1. Preflight: check the cost counter before training anything

This takes seconds and is check 3 of the reproduction gate (`docs/DATA.md`). If it
disagrees, stop — every downstream number is meaningless.

```bash
uv run python -c "
from harness.cost import count_cost
from harness.reference import a1_standard, a2_nano, a2_standard
for name, factory in [('a2_standard', a2_standard), ('a2_nano', a2_nano), ('a1_standard', a1_standard)]:
    r = count_cost(factory(), device='cpu')
    print(f'{name:14s} macs={r.macs_per_sample:9.1f} params={r.params:6d} elementwise={r.elementwise_per_sample:7.1f} linear={r.is_linear}')
"
```

Expected, agreeing with three independent derivations and a real deployed `.nam`:

| | MACs/sample | params |
|---|---|---|
| `a2_standard` | 11,776 | 12,145 |
| `a2_nano` | 1,731 | 1,870 |
| `a1_standard` | 13,320 | 13,801 |

---

## 2. Data preparation (one-time, ~2 GB)

```bash
uv run prepare.py --sources sources.json
```

Fetches 4 panel + 4 holdout captures, estimates each pair's reamping latency by
cross-correlation, normalizes to −18 dBFS RMS, and writes
`~/.cache/nam-autoresearch/panel.json`. Archives are cached, so a re-run is cheap.

Watch the reported delays. They should be small and stable (tens to a few hundred
samples). A delay pinned at `0` or at the `4096` search ceiling means the estimate
failed, and misalignment inflates ESR in a way indistinguishable from a bad
architecture — that is a whole night wasted chasing a latency bug.

Read `docs/DATA.md` before publishing anything derived from these captures. The A2
paper captures carry **no stated license**, and ToneTwist is **CC-BY-NC-4.0**.

---

## 3. The baseline sweep

`train.py` has no CLI by design — it is the file the agent rewrites, so its knobs are
module constants. The sweep therefore edits it in place and restores it afterwards.

```bash
mkdir -p runs

for arch in a2_standard a2_nano a1_standard lstm; do
  sed -i "s/^BASELINE = .*/BASELINE = \"$arch\"/" train.py
  echo "=== $arch ==="
  PYTHONUNBUFFERED=1 uv run train.py > "runs/${arch}_seed0.log" 2>&1
  tail -20 "runs/${arch}_seed0.log"
done

git checkout train.py    # restore BASELINE = "a2_standard"
```

Each run trains 8 fresh models (4 panel + 4 holdout, one per capture) inside a
600-second total budget — 75 s each — then evaluates and costs the result. Expect
12–15 minutes per architecture, so about an hour for the four.

`git checkout train.py` matters: leaving `BASELINE` on `lstm` would make every later
run silently train the wrong thing.

### Then the candidate

`wh_ssm` is the first novel architecture and the reason the streaming-cost machinery
exists. It is not a baseline — it is the first thing the baselines are for.

```bash
sed -i 's/^BASELINE = .*/BASELINE = "wh_ssm"/' train.py
PYTHONUNBUFFERED=1 uv run train.py > runs/wh_ssm_seed0.log 2>&1
git checkout train.py
```

Its report line should read `cost_form: streaming (equiv esr ...)` with the
equivalence around `1e-14`. If it says `as trained`, the streaming form was not used
and the 10,160 figure does not apply to that row. Note this run is slower to score:
verifying the streaming form costs about 30 s on top of training.

---

## 4. The noise floor

Two more seeds of the incumbent. This is the single most important number in the
sweep — without it the loop banks run-to-run variance for fifty iterations while the
log shows steady progress.

```bash
for seed in 1 2; do
  sed -i "s/^SEED = .*/SEED = $seed/" train.py
  PYTHONUNBUFFERED=1 uv run train.py > "runs/a2_standard_seed${seed}.log" 2>&1
done

git checkout train.py

uv run python tools/collect.py --noise-floor runs/a2_standard_seed*.log
```

Output:

```
a2_standard_seed0.log                esr=0.004210
a2_standard_seed1.log                esr=0.004455
a2_standard_seed2.log                esr=0.004102

noise_floor (max-min over 3 runs): 0.000353
relative to mean: 8.29%
```

(Illustrative numbers, not measured.) Spread rather than standard deviation — with
three seeds, spread is the honest statement of "differences this small are not
evidence", and it claims no distribution the sample size cannot support.

**Read the relative figure carefully.** If the floor lands near 10% of the score,
then any candidate improving ESR by less than that is unprovable at this budget, and
most architectural tweaks land well inside it. That is a real finding about the
experiment's resolution, not a nuisance — if it comes out that large, the honest
responses are more seeds per config or a longer per-run budget, not a smaller margin.

---

## 5. Record the runs

```bash
uv run python tools/collect.py runs/*.log
column -t -s$'\t' results.tsv
```

Logs without a complete results block (a run killed by the timeout) are skipped with a
warning rather than recorded as results. `status` stays as the harness reported it —
`ok` or `invalid` — because `keep`/`discard` is `harness.verdict.decide`'s call and
needs the incumbent and the floor.

---

## 6. The acceptance gate

There is **no published per-capture A2 ESR** to check against — TONE3000 released only
MUSHRA listener ratings, and the Slimmable paper's curve describes A1-shaped models.
So the gate is internal consistency (`docs/DATA.md`), and all four must hold:

1. **`a2_standard` beats `a1_standard`** — at lower compute (11,776 vs 13,320
   MACs/sample). This is A2's entire claim. If a result this lopsided does not
   reproduce, the harness is wrong.
2. **`a2_standard` beats `a2_nano`** by a clear margin.
3. **Cost figures match** the table in step 1.
4. **ESR is in a sane absolute range** — order 1e-3 to 1e-2 on these captures. 1e-1
   means something is broken, most likely alignment.

`lstm` is not part of the gate. At 51 MACs/sample it marks the bottom of the space
rather than competing.

If any of 1–4 fails, nothing downstream means anything, including any `wh_ssm` result
from the same sweep. Fix the harness first.

---

## 7. Optional: calibrate the runtime cap

Until this runs, `predicted_us` prints as `ADVISORY-uncalibrated` and only the MAC and
elementwise caps bind. Coefficients are machine-specific and do not transfer.

```bash
cd /path/to/modulus && cargo build -p test-harness --release --bin nam_diag
cd -
uv run python tools/calibrate.py --profile my-workstation --modulus /path/to/modulus
```

Then set `CALIBRATION_PROFILE = "my-workstation"` in `harness/constants.py` and
re-run one baseline to confirm the line changes to `gating`.

The fit refuses to gate while under-identified, so a thin design matrix leaves the cap
advisory rather than silently applying a wrong model. Depth must vary independently of
width or the per-convolution coefficient cannot be separated from the intercept — the
degeneracy the bundled `modulus-i9-14900f` profile suffers from.

---

## What this unlocks

With an incumbent ESR and a measured floor in `results.tsv`, `harness.verdict.decide`
has both inputs it needs, and the loop in `program.md` can run unattended. Before that
it cannot: every candidate would be compared against nothing, with no threshold for
what counts as evidence.
