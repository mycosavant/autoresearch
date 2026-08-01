"""
Calibrate the runtime cost model for *this* machine.

Cost-model coefficients encode a specific CPU's arithmetic throughput and per-call
overhead, so they do not transfer between machines. Run this on whatever box will
host the research loop (and again on any deployment target you care about).

What it does:

1. Generates WaveNet variants spanning depth and width (``make_calibration_models``).
   Depth must vary independently of width, or the per-convolution coefficient cannot
   be separated from the intercept -- the exact degeneracy the ``modulus-i9-14900f``
   profile suffers from.
2. Benchmarks each with modulus's ``nam_diag``.
3. Fits, checks identifiability, and writes ``harness/calibration/<profile>.json``.

Requires a built modulus checkout::

    cd /workspace/modulus && cargo build -p test-harness --release --bin nam_diag

Note the binary is ``nam_diag`` with an underscore; modulus's own docs say
``nam-diag``, which does not resolve.

Usage::

    python tools/calibrate.py --profile my-workstation
    # then set CALIBRATION_PROFILE = "my-workstation" in harness/constants.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.cost_model import CALIBRATION_DIR, Measurement, fit  # noqa: E402
from tools.make_calibration_models import main as generate  # noqa: E402

_MEAN_RE = re.compile(r"mean block:\s*([0-9.]+)")


def benchmark(binary: Path, model: Path, wav: Path, block: int, repeats: int) -> float:
    """Return mean microseconds per block, or raise if the tool did not report one."""
    proc = subprocess.run(
        [
            str(binary),
            "--model", str(model),
            "--input", str(wav),
            "--block-size", str(block),
            "--sample-rate", "48000",
            "--repeats", str(repeats),
        ],
        capture_output=True,
        text=True,
    )
    # nam_diag writes its human-readable report to stderr.
    match = _MEAN_RE.search(proc.stderr) or _MEAN_RE.search(proc.stdout)
    if not match:
        raise RuntimeError(
            f"No timing in nam_diag output for {model.name}.\n"
            f"stderr tail: {proc.stderr[-400:]}"
        )
    return float(match.group(1))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, help="Name for this machine's profile.")
    parser.add_argument("--modulus", type=Path, default=Path("/workspace/modulus"))
    parser.add_argument("--work", type=Path, default=Path("/tmp/calib"))
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument(
        "--invocations",
        type=int,
        default=3,
        help="Separate runs per model; the median is taken, since a shared CPU "
             "produces occasional outliers that a single run cannot detect.",
    )
    parser.add_argument("--machine", default="", help="Free-text machine description.")
    args = parser.parse_args(argv)

    binary = args.modulus / "target" / "release" / "nam_diag"
    wav = args.modulus / "crates" / "test-harness" / "tests" / "fixtures" / "input.wav"
    for path, hint in ((binary, "cargo build -p test-harness --release --bin nam_diag"),
                       (wav, "check the modulus checkout")):
        if not path.exists():
            print(f"Missing {path}\n  -> {hint}")
            return 1

    print("generating calibration models...\n")
    generate(["--out", str(args.work)])

    manifest = {m["name"]: m for m in json.loads((args.work / "manifest.json").read_text())}

    print(f"\nbenchmarking (block={args.block_size}, repeats={args.repeats}, "
          f"{args.invocations} invocations each)\n")
    measurements: List[Measurement] = []
    records: List[Dict] = []

    for name, entry in manifest.items():
        samples = [
            benchmark(binary, args.work / entry["file"], wav, args.block_size, args.repeats)
            for _ in range(args.invocations)
        ]
        median = statistics.median(samples)
        spread = (max(samples) - min(samples)) / median if median else 0.0
        per_sample = median / args.block_size

        print(f"  {name:20s} {median:8.1f} us/block  (spread {spread:5.1%})")
        measurements.append(Measurement(name, entry["features"], per_sample))
        records.append(
            {
                "name": name,
                "features": entry["features"],
                "microseconds_per_sample": per_sample,
                "n_repeats": args.invocations,
                "spread_rel": round(spread, 4),
            }
        )

    model = fit(measurements, provenance=args.profile)
    print()
    print(model.explain())

    if not model.is_adequate:
        print(
            "\nWARNING: this calibration is not adequate to gate on. It will be "
            "written and reported, but the runtime cap will stay advisory."
        )

    out = CALIBRATION_DIR / f"{args.profile}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "machine": args.machine or "unspecified",
                "tool": (
                    f"modulus nam_diag, block {args.block_size}, sr 48000, "
                    f"--repeats {args.repeats}, median of {args.invocations} invocations"
                ),
                "measurements": records,
            },
            indent=2,
        )
    )
    print(f"\nwrote {out}")
    print(f'set CALIBRATION_PROFILE = "{args.profile}" in harness/constants.py to use it')
    return 0


if __name__ == "__main__":
    sys.exit(main())
