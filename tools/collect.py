"""
Turn run logs into ``results.tsv`` rows, and measure the ESR noise floor.

``train.py`` prints a results block; nothing until now read it back. During the
autonomous loop the agent transcribes it by hand, which is fine for one run at a
time and useless for a baseline sweep of seven.

Usage::

    # append rows for finished runs
    python tools/collect.py runs/*.log

    # print rows without touching results.tsv
    python tools/collect.py --dry-run runs/*.log

    # the noise floor, from repeated runs of one identical config
    python tools/collect.py --noise-floor runs/a2_standard_seed*.log

The ``status`` column is deliberately left as the harness reported it (``ok`` or
``invalid``) rather than being translated to ``keep``/``discard``. That decision
belongs to :func:`harness.verdict.decide`, which needs the incumbent and the noise
floor -- neither of which a log parser has any business guessing at.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.constants import RESULTS_COLUMNS, RESULTS_PATH  # noqa: E402

# ``harness.verdict`` is imported lazily, in the one branch that needs it. It pulls
# in ``harness.cost`` and therefore torch, and parsing text should not require a CUDA
# install -- collecting rows from logs copied off the research box is a normal thing
# to want to do. The floor itself is still the harness's definition rather than a
# second copy of it, which is the part that matters.

#: Lines in the results block look like ``key:   value``, padded for reading.
_FIELD = re.compile(r"^([a-z_]+):\s+(.*?)\s*$")

#: Fields that must be present for a log to count as a finished run. A run killed
#: by the timeout leaves a truncated block, and silently emitting a row for it
#: would put a half-trained model in the log looking like a real result.
_REQUIRED = ("esr", "mrstft", "macs_per_sample", "params", "status")


def parse(log: Path) -> Optional[Dict[str, str]]:
    """Extract the results block from one log, or ``None`` if it never finished."""
    fields: Dict[str, str] = {}
    in_block = False
    for line in log.read_text(errors="replace").splitlines():
        if line.strip() == "---":
            in_block = True
            continue
        if line.startswith("--- per capture"):
            break
        if in_block:
            match = _FIELD.match(line)
            if match:
                fields[match.group(1)] = match.group(2)

    missing = [f for f in _REQUIRED if f not in fields]
    if missing:
        return None
    return fields


def _commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def row(fields: Dict[str, str], description: str, commit: str) -> str:
    values = {
        "commit": commit,
        "esr": fields.get("esr", ""),
        "mrstft": fields.get("mrstft", ""),
        "esr_holdout": fields.get("esr_holdout", ""),
        "macs_per_sample": fields.get("macs_per_sample", ""),
        "params": fields.get("params", ""),
        "rtf": fields.get("rtf", "n/a"),
        "status": fields.get("status", ""),
        "description": description,
    }
    return "\t".join(values[c] for c in RESULTS_COLUMNS)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="Print rows, write nothing.")
    parser.add_argument(
        "--noise-floor",
        action="store_true",
        help="Report the ESR spread across the given logs instead of emitting rows.",
    )
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    args = parser.parse_args(argv)

    parsed = [(log, parse(log)) for log in args.logs]
    for log, fields in parsed:
        if fields is None:
            print(f"skipped (no complete results block): {log}", file=sys.stderr)

    good = [(log, f) for log, f in parsed if f is not None]
    if not good:
        print("No finished runs found.", file=sys.stderr)
        return 1

    if args.noise_floor:
        from harness.verdict import noise_floor

        scores = [float(f["esr"]) for _, f in good]
        if len(scores) < 2:
            print("Need at least 2 finished runs to estimate a floor.", file=sys.stderr)
            return 1
        floor = noise_floor(scores)
        for (log, _), score in zip(good, scores):
            print(f"{log.name:36s} esr={score:.6f}")
        print(f"\nnoise_floor (max-min over {len(scores)} runs): {floor:.6f}")
        print(f"relative to mean: {floor / (sum(scores) / len(scores)):.2%}")
        print(
            "\nAn improvement must exceed this to be kept "
            "(harness.constants.KEEP_MARGIN_FACTOR)."
        )
        return 0

    commit = _commit()
    rows = [row(f, log.stem, commit) for log, f in good]

    if args.dry_run:
        print("\t".join(RESULTS_COLUMNS))
        print("\n".join(rows))
        return 0

    header = "\t".join(RESULTS_COLUMNS)
    existing = args.results.read_text() if args.results.is_file() else ""
    with args.results.open("a") as fh:
        if not existing.strip():
            fh.write(header + "\n")
        for r in rows:
            fh.write(r + "\n")
    print(f"appended {len(rows)} row(s) to {args.results}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
