"""
One-time data preparation. READ-ONLY: not to be modified by the research agent.

Ingests dry/wet capture pairs, aligns and normalizes them, and writes the panel
manifest that every experiment then loads identically.

Alignment is the part that earns its keep. A capture pair recorded through a
reamping loop carries an unknown latency of anywhere from a handful to a few hundred
samples. A few samples of misalignment inflates ESR by orders of magnitude, and it
does so in a way that is indistinguishable from a bad architecture -- an entire
night's search can be spent chasing a delay. So the delay is estimated once per
capture by cross-correlation, frozen into the manifest, and applied identically on
every load.

Usage
-----
    uv run prepare.py --sources sources.json

``sources.json`` describes where the captures live::

    {
      "panel": [
        {"name": "fender_deluxe_clean", "category": "clean",
         "input": "/data/fender/input.wav", "target": "/data/fender/target.wav"}
      ],
      "holdout": [...]
    }

Paths may be local files or ``http(s)://`` URLs.
"""

from __future__ import annotations

import argparse as _argparse
import json as _json
import shutil as _shutil
import sys as _sys
import urllib.request as _urlreq
from pathlib import Path as _Path
from typing import Dict, List, Optional, Tuple

import numpy as _np

from harness.constants import DATA_DIR, MANIFEST_PATH, SAMPLE_RATE

#: Maximum dry/wet latency searched, in samples. 4096 @ 48 kHz is ~85 ms, which
#: comfortably covers reamping interfaces, converter latency and plugin delay
#: compensation errors.
MAX_DELAY_SEARCH = 4096

#: Seconds used for delay estimation. A short window keeps the correlation cheap;
#: taking it from the middle avoids fade-ins and leading silence, which correlate
#: poorly and give unstable estimates.
DELAY_ESTIMATE_SECONDS = 20.0

#: Peak level captures are normalized to, leaving headroom.
TARGET_PEAK = 0.95


def _log(msg: str) -> None:
    print(msg, flush=True)


def estimate_delay(x: _np.ndarray, y: _np.ndarray, max_delay: int = MAX_DELAY_SEARCH) -> int:
    """Estimate how many samples ``y`` lags behind ``x``, by cross-correlation.

    Positive result means the wet signal is late, which is the normal case.

    Uses an FFT cross-correlation over a mid-signal excerpt. Both signals are
    mean-removed and energy-normalized first, so the estimate depends on waveform
    shape rather than on the amplifier's gain -- which matters, because a high-gain
    capture can be 30 dB louder than its input.
    """
    n = int(min(len(x), len(y), DELAY_ESTIMATE_SECONDS * SAMPLE_RATE))
    if n <= max_delay * 2:
        raise ValueError(f"Capture too short ({n} samples) to estimate delay reliably.")

    mid = min(len(x), len(y)) // 2
    start = max(0, mid - n // 2)
    a = x[start:start + n].astype(_np.float64)
    b = y[start:start + n].astype(_np.float64)

    a -= a.mean()
    b -= b.mean()
    a_norm, b_norm = _np.linalg.norm(a), _np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        raise ValueError("Silent segment; cannot estimate delay.")
    a /= a_norm
    b /= b_norm

    size = 1 << int(_np.ceil(_np.log2(n + max_delay)) + 1)
    corr = _np.fft.irfft(_np.fft.rfft(b, size) * _np.conj(_np.fft.rfft(a, size)), size)

    # Only non-negative lags: the wet signal cannot precede the dry one.
    lags = corr[: max_delay + 1]
    return int(_np.argmax(_np.abs(lags)))


def _fetch(src: str, dest: _Path, cache: Optional[_Path] = None) -> _Path:
    """Copy or download ``src`` to ``dest``.

    Accepts a local path, an ``http(s)`` URL, or ``<zip-url>#<path/inside.zip>``.
    The zip form exists because the ToneTwist records are distributed as one archive
    per device; downloading a 1.6 GB zip once and pulling two files out of it beats
    asking every user to unpack by hand.
    """
    if "#" in src and src.startswith(("http://", "https://")):
        archive_url, inner = src.split("#", 1)
        archive = _download_cached(archive_url, cache or dest.parent)
        import zipfile

        with zipfile.ZipFile(archive) as zf:
            try:
                member = zf.getinfo(inner)
            except KeyError:
                candidates = [n for n in zf.namelist() if n.endswith(inner)]
                if len(candidates) != 1:
                    raise FileNotFoundError(
                        f"{inner!r} not found in {archive_url} "
                        f"({len(candidates)} fuzzy matches)"
                    )
                member = zf.getinfo(candidates[0])
            with zf.open(member) as fh, dest.open("wb") as out:
                _shutil.copyfileobj(fh, out)
        return dest

    if src.startswith(("http://", "https://")):
        _log(f"    downloading {src}")
        with _urlreq.urlopen(src) as response, dest.open("wb") as fh:
            _shutil.copyfileobj(response, fh)
        return dest

    source = _Path(src).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Capture file not found: {source}")
    _shutil.copyfile(source, dest)
    return dest


def _download_cached(url: str, cache_dir: _Path) -> _Path:
    """Download ``url`` once and reuse it. Archives here are up to ~3 GB."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    name = url.rstrip("/").split("/")[-1].split("?")[0] or "archive.zip"
    path = cache_dir / f"_cache_{name}"
    if path.is_file():
        return path
    _log(f"    downloading archive {url}")
    tmp = path.with_suffix(path.suffix + ".part")
    with _urlreq.urlopen(url) as response, tmp.open("wb") as fh:
        _shutil.copyfileobj(response, fh)
    tmp.rename(path)
    return path


def _load_and_normalize(path: _Path) -> _np.ndarray:
    from nam.data import wav_to_np

    x = wav_to_np(path, rate=SAMPLE_RATE)
    x = _np.asarray(x, dtype=_np.float32).reshape(-1)
    peak = float(_np.abs(x).max())
    if peak == 0.0:
        raise ValueError(f"{path} is silent.")
    return x


def _prepare_entry(entry: Dict, out_dir: _Path) -> Dict:
    from nam.data import np_to_wav

    name = entry["name"]
    _log(f"  {name}")
    dest = out_dir / name
    dest.mkdir(parents=True, exist_ok=True)

    raw_in = _fetch(entry["input"], dest / "_raw_input.wav")
    raw_out = _fetch(entry["target"], dest / "_raw_target.wav")

    x = _load_and_normalize(raw_in)
    y = _load_and_normalize(raw_out)

    delay = estimate_delay(x, y)
    _log(f"    delay={delay} samples ({1000 * delay / SAMPLE_RATE:.2f} ms)")

    # Normalize levels *after* alignment. Peak-normalizing is deliberate rather than
    # RMS: ESR is energy-relative already, and a peak-normalized target keeps the
    # nonlinearity operating at the level it was captured at.
    x = x / _np.abs(x).max() * TARGET_PEAK
    y = y / _np.abs(y).max() * TARGET_PEAK

    np_to_wav(x, dest / "input.wav", rate=SAMPLE_RATE)
    np_to_wav(y, dest / "target.wav", rate=SAMPLE_RATE)
    raw_in.unlink(missing_ok=True)
    raw_out.unlink(missing_ok=True)

    seconds = len(x) / SAMPLE_RATE
    _log(f"    {seconds:.1f}s @ {SAMPLE_RATE} Hz")

    return {
        "name": name,
        "category": entry.get("category", "unknown"),
        "dir": name,
        "input": "input.wav",
        "target": "target.wav",
        "delay": delay,
        "seconds": round(seconds, 2),
        "source": {"input": entry["input"], "target": entry["target"]},
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = _argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        type=_Path,
        default=_Path("sources.json"),
        help="JSON describing the capture panel and holdout.",
    )
    parser.add_argument("--data-dir", type=_Path, default=DATA_DIR)
    parser.add_argument("--manifest", type=_Path, default=MANIFEST_PATH)
    args = parser.parse_args(argv)

    if not args.sources.is_file():
        _log(
            f"No sources file at {args.sources}.\n\n"
            "Create one describing your captures, e.g.:\n\n"
            '{\n'
            '  "panel": [\n'
            '    {"name": "fender_deluxe_clean", "category": "clean",\n'
            '     "input": "/data/fender/input.wav",\n'
            '     "target": "/data/fender/target.wav"}\n'
            '  ],\n'
            '  "holdout": []\n'
            "}\n\n"
            "See docs/DATA.md for the recommended panel."
        )
        return 1

    sources = _json.loads(args.sources.read_text())
    args.data_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {"sample_rate": SAMPLE_RATE}
    for split in ("panel", "holdout"):
        entries = sources.get(split, [])
        if entries:
            _log(f"{split}:")
        manifest[split] = [_prepare_entry(e, args.data_dir) for e in entries]

    if not manifest["panel"]:
        _log("ERROR: the panel is empty; at least one capture is required.")
        return 1

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(_json.dumps(manifest, indent=2))
    _log(f"\nwrote {args.manifest}")
    _log(f"panel={len(manifest['panel'])} holdout={len(manifest['holdout'])}")
    return 0


if __name__ == "__main__":
    _sys.exit(main())
