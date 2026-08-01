"""End-to-end smoke test on synthetic captures. Not part of the research loop."""
import json, sys, tempfile
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nam.data import np_to_wav
from harness.data import load_panel
from harness.runner import TimeBudget, report
import train as T

SR = 48000

def synth(seed, n=SR*30):
    """Dry = filtered noise; wet = a soft-clipped, filtered version of it."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n).astype(np.float32)
    # crude one-pole lowpass so it isn't white
    b = 0.0
    for i in range(n):
        b = 0.85*b + 0.15*x[i]; x[i] = b
    x /= np.abs(x).max()
    drive = 3.0 + seed
    y = np.tanh(drive * x)
    c = 0.0
    for i in range(n):
        c = 0.7*c + 0.3*y[i]; y[i] = c
    y = (y / np.abs(y).max() * 0.7).astype(np.float32)
    return x, y

def main():
    root = Path(tempfile.mkdtemp(prefix="namtest-"))
    data = root / "captures"; data.mkdir()
    entries = []
    for i, (name, cat) in enumerate([("amp_clean","clean"),("amp_crunch","crunch"),
                                     ("amp_hg","high-gain"),("pedal_od","pedal"),
                                     ("holdout_amp","holdout")]):
        d = data / name; d.mkdir()
        x, y = synth(i)
        np_to_wav(x, d/"input.wav", rate=SR)
        np_to_wav(y, d/"target.wav", rate=SR)
        entries.append({"name": name, "category": cat, "dir": name, "delay": 0})
    manifest = {"sample_rate": SR, "panel": entries[:4], "holdout": entries[4:]}
    mpath = root / "panel.json"; mpath.write_text(json.dumps(manifest, indent=2))

    panel = load_panel(mpath, data, val_seconds=3.0)
    print(panel)
    for c in panel.panel:
        print(f"  {c.name:14s} train={c.train_samples} val={c.val_samples}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T.NY = 512          # keep the smoke test fast on CPU
    T.BATCH_SIZE = 4
    captures = list(panel.panel) + list(panel.holdout)
    budget = TimeBudget(total_seconds=10.0, n_captures=len(captures))
    models = {c.name: T.train_one(c, budget, device) for c in captures}
    result = report(models, panel, training_seconds=budget.elapsed_total, device=device)
    print(f"\nSMOKE OK status={result.status} esr={result.metrics.esr:.6f} "
          f"macs={result.cost.macs_per_sample:.0f}")

main()
