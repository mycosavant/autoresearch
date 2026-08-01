"""
Capture loading and splits. READ-ONLY: not to be modified by the research agent.

The panel is deliberately plural. A single capture would let the search overfit the
*architecture* to one amp's nonlinearity -- a model tuned until it excels at one
high-gain head tells you nothing about whether it is a better architecture. So every
experiment trains on a panel of captures spanning clean / crunch / high-gain / pedal
and is scored on the mean.

A separate holdout of amps is loaded but never trained on. It is reported, never
optimized against: tuning on it would convert the only unbiased generalization signal
in the setup into just another training target.

Splits are by *time*, not by random windows, following NAM's own convention: the tail
of each capture is validation. Random-window splits would leak, because neighbouring
windows of the same recording are near-duplicates.
"""

from __future__ import annotations

import json as _json
from dataclasses import dataclass as _dataclass
from pathlib import Path as _Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch as _torch

from .constants import DATA_DIR, MANIFEST_PATH, SAMPLE_RATE

__all__ = ["Capture", "Panel", "load_panel", "manifest_exists"]

#: Seconds of each capture reserved for validation, taken from the tail.
#: Matches NAM's default single-pair config (stop_seconds = -9.0).
VALIDATION_SECONDS = 9.0


class MissingDataError(FileNotFoundError):
    """Raised when the capture cache has not been prepared."""


@_dataclass
class Capture:
    """One dry/wet pair, split into train and validation segments."""

    name: str
    category: str  # clean | crunch | high-gain | pedal
    x_train: _torch.Tensor  # (N,) dry input
    y_train: _torch.Tensor  # (N,) wet target
    x_val: _torch.Tensor
    y_val: _torch.Tensor

    @property
    def train_samples(self) -> int:
        return int(self.x_train.numel())

    @property
    def val_samples(self) -> int:
        return int(self.x_val.numel())

    def to(self, device) -> "Capture":
        return Capture(
            name=self.name,
            category=self.category,
            x_train=self.x_train.to(device),
            y_train=self.y_train.to(device),
            x_val=self.x_val.to(device),
            y_val=self.y_val.to(device),
        )


@_dataclass
class Panel:
    """The scored panel plus the never-trained-on holdout."""

    panel: List[Capture]
    holdout: List[Capture]

    def to(self, device) -> "Panel":
        return Panel([c.to(device) for c in self.panel], [c.to(device) for c in self.holdout])

    def __repr__(self) -> str:  # pragma: no cover - display only
        names = ", ".join(f"{c.name}({c.category})" for c in self.panel)
        return f"Panel(panel=[{names}], holdout={len(self.holdout)} captures)"


def manifest_exists() -> bool:
    return MANIFEST_PATH.is_file()


def _load_pair(entry: Dict, data_dir: _Path) -> Tuple[_torch.Tensor, _torch.Tensor]:
    """Load one dry/wet pair, applying the manifest's alignment delay."""
    from nam.data import wav_to_tensor  # lazy: pulls torch + NAM

    base = data_dir / entry["dir"]
    x = wav_to_tensor(base / entry.get("input", "input.wav"), rate=SAMPLE_RATE)
    y = wav_to_tensor(base / entry.get("target", "target.wav"), rate=SAMPLE_RATE)

    # Alignment matters more than it looks: a few samples of latency between dry and
    # wet inflates ESR enormously and would be indistinguishable from a bad
    # architecture. prepare.py estimates this per capture and freezes it here.
    delay = int(entry.get("delay", 0))
    if delay > 0:
        x, y = x[delay:], y[:-delay]
    elif delay < 0:
        x, y = x[:delay], y[-delay:]

    n = min(x.numel(), y.numel())
    return x[:n].float(), y[:n].float()


def _split(x: _torch.Tensor, y: _torch.Tensor, val_seconds: float):
    n_val = int(val_seconds * SAMPLE_RATE)
    if x.numel() <= n_val * 2:
        raise ValueError(
            f"Capture too short: {x.numel()} samples with {n_val} reserved for "
            f"validation leaves too little to train on."
        )
    return x[:-n_val], y[:-n_val], x[-n_val:], y[-n_val:]


def load_panel(
    manifest_path: Optional[_Path] = None,
    data_dir: Optional[_Path] = None,
    *,
    val_seconds: float = VALIDATION_SECONDS,
) -> Panel:
    """Load the capture panel and holdout described by the manifest.

    :raises MissingDataError: if the cache has not been prepared.
    """
    manifest_path = manifest_path or MANIFEST_PATH
    data_dir = data_dir or DATA_DIR

    if not manifest_path.is_file():
        raise MissingDataError(
            f"No capture manifest at {manifest_path}. Run `uv run prepare.py` first."
        )

    manifest = _json.loads(manifest_path.read_text())

    def build(entries: Sequence[Dict]) -> List[Capture]:
        out = []
        for entry in entries:
            x, y = _load_pair(entry, data_dir)
            xt, yt, xv, yv = _split(x, y, val_seconds)
            out.append(
                Capture(
                    name=entry["name"],
                    category=entry.get("category", "unknown"),
                    x_train=xt,
                    y_train=yt,
                    x_val=xv,
                    y_val=yv,
                )
            )
        return out

    panel = build(manifest.get("panel", []))
    if not panel:
        raise ValueError(f"Manifest {manifest_path} defines an empty panel.")

    return Panel(panel=panel, holdout=build(manifest.get("holdout", [])))
