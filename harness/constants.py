"""
Fixed experimental constants. READ-ONLY: not to be modified by the research agent.

Everything here defines what "an experiment" means. Changing any of it invalidates
comparison against every result already in the log, which is why it lives outside
the file the agent edits.
"""

from __future__ import annotations

from pathlib import Path as _Path

# --------------------------------------------------------------------------------
# Audio
# --------------------------------------------------------------------------------

#: NAM's operating sample rate. Captures are resampled to this on ingest.
SAMPLE_RATE = 48_000

#: Pre-emphasis coefficient for the headline ESR metric.
#: NAM's own training core uses 0.85 (nam/train/core.py); the originating paper
#: (mdpi.com/2076-3417/10/3/766) uses 0.95. We follow NAM so that numbers out of
#: this harness are directly comparable to published NAM/A2 results.
PRE_EMPH_COEF = 0.85

#: Multi-resolution STFT resolutions. Matches the auraloss defaults vendored in NAM.
MRSTFT_FFT_SIZES = (1024, 2048, 512)
MRSTFT_HOP_SIZES = (120, 240, 50)
MRSTFT_WIN_LENGTHS = (600, 1200, 240)

# --------------------------------------------------------------------------------
# Experiment budget
# --------------------------------------------------------------------------------

#: Total wall-clock training budget for ONE experiment, in seconds, excluding
#: startup/compile/eval. Split evenly across the capture panel. At 600s over a
#: 4-capture panel that is 150s per capture, ~6 experiments/hour.
EXPERIMENT_SECONDS = 600.0

#: Hard ceiling on total experiment wall-clock, including startup and eval. A run
#: exceeding this is killed and recorded as a failure.
EXPERIMENT_TIMEOUT_SECONDS = 1_800.0

# --------------------------------------------------------------------------------
# Compute cap
# --------------------------------------------------------------------------------

#: The compute cap is DERIVED at runtime from the frozen reference architecture in
#: harness/reference.py (A2 standard), not stored as a magic number. There is
#: therefore no budget file to tamper with: the cap is whatever A2 standard costs.
#:
#: Fractional slack allowed above the reference cost. Small and deliberate: the
#: question this harness asks is "can anything beat A2 at A2's compute", so a
#: candidate that needs materially more compute is answering a different question.
MAC_BUDGET_TOLERANCE = 0.02

#: Elementwise ops are not MACs but are not free either. A candidate may not exceed
#: the reference's elementwise count by more than this factor, which stops an
#: architecture from moving its work into giant pointwise gating to dodge the cap.
ELEMENTWISE_BUDGET_FACTOR = 3.0

# --------------------------------------------------------------------------------
# Decision rule
# --------------------------------------------------------------------------------

#: Number of seeds the baseline is run with to establish the ESR noise floor.
NOISE_FLOOR_SEEDS = 3

#: An improvement must exceed (noise_floor * this) to be kept. Guards against the
#: loop banking run-to-run variance for dozens of iterations and drifting.
KEEP_MARGIN_FACTOR = 1.0

# --------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------

CACHE_DIR = _Path.home() / ".cache" / "nam-autoresearch"
DATA_DIR = CACHE_DIR / "captures"
MANIFEST_PATH = CACHE_DIR / "panel.json"
RESULTS_PATH = _Path(__file__).resolve().parent.parent / "results.tsv"

#: results.tsv schema. Extends Karpathy's 5-column log with the cost axis, the
#: generalization holdout, and the metric-gaming guard.
RESULTS_COLUMNS = (
    "commit",
    "esr",            # headline: pre-emphasis ESR, mean over panel
    "mrstft",         # perceptual cross-check
    "esr_holdout",    # amps never trained on
    "macs_per_sample",
    "params",
    "rtf",            # measured real-time factor, or "n/a"
    "status",         # keep | discard | crash | invalid
    "description",
)
