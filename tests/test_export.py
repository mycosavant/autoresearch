"""Deployment-form export tests.

The point of these is to fail *here* rather than in Rust. A backend written against
``tools/export_wh_ssm.py``'s output can only be correct if that output is complete
and unambiguous, so the central test re-implements the whole cascade from the JSON
alone -- scalar arithmetic, no torch, no access to the original module -- and checks
it against the golden vectors. If that passes, every remaining parity failure is a
bug in the backend rather than a hole in the format.

Re-implementing rather than round-tripping through torch is deliberate. A
round-trip would share layout assumptions with the exporter and agree with it even
when both are wrong; the classic case is a transposed weight matrix, which survives
any test that reads it back the same way it was written.
"""

import math

import numpy as np
import pytest
import torch

import train as T
from tools.export_wh_ssm import export_model


@pytest.fixture(scope="module")
def exported():
    torch.manual_seed(0)
    model = T.WienerHammersteinSSM().eval()
    return model, export_model(model, golden_samples=512)


def _run_bank(stage, x):
    """Scalar streaming recurrence, straight from the exported fields."""
    nb, in_ch, out_ch = stage["n_blocks"], stage["in_ch"], stage["out_ch"]
    b = np.asarray(stage["b"], dtype=np.float64).reshape(nb, 2, in_ch)
    a00 = np.asarray(stage["a00"], dtype=np.float64)
    a01 = np.asarray(stage["a01"], dtype=np.float64)
    a10 = np.asarray(stage["a10"], dtype=np.float64)
    a11 = np.asarray(stage["a11"], dtype=np.float64)
    cw = np.asarray(stage["c_w"], dtype=np.float64).reshape(out_ch, 2 * nb)
    cb = np.asarray(stage["c_b"], dtype=np.float64)

    n = x.shape[1]
    h0 = np.zeros(nb)
    h1 = np.zeros(nb)
    out = np.zeros((out_ch, n))
    for t in range(n):
        xt = x[:, t]
        u0 = b[:, 0, :] @ xt
        u1 = b[:, 1, :] @ xt
        n0 = a00 * h0 + a01 * h1 + u0
        n1 = a10 * h0 + a11 * h1 + u1
        h0, h1 = n0, n1
        state = np.empty(2 * nb)
        state[0::2] = h0
        state[1::2] = h1
        out[:, t] = cw @ state + cb
    return out


def _run_nonlinearity(stage, x):
    ch, hid = stage["channels"], stage["hidden"]
    w1 = np.asarray(stage["w1"], dtype=np.float64).reshape(hid, ch)
    b1 = np.asarray(stage["b1"], dtype=np.float64)
    w2 = np.asarray(stage["w2"], dtype=np.float64).reshape(ch, hid)
    b2 = np.asarray(stage["b2"], dtype=np.float64)
    return w2 @ np.tanh(w1 @ x + b1[:, None]) + b2[:, None]


def _run_cascade(payload, x):
    z = x.reshape(1, -1).astype(np.float64)
    for stage in payload["stages"]:
        z = _run_bank(stage, z) if stage["kind"] == "bank" else _run_nonlinearity(stage, z)
    return z.reshape(-1)


def test_reimplementation_from_the_file_alone_matches_the_golden_output(exported):
    """The load-bearing test: the format is sufficient to reproduce the model."""
    _, payload = exported
    x = np.asarray(payload["golden"]["input"], dtype=np.float64)
    want = np.asarray(payload["golden"]["output"], dtype=np.float64)

    got = _run_cascade(payload, x)

    # float32 forward in torch against float64 here, over a cascade of three
    # recurrences -- so this is float32 epsilon accumulated, not a modelling
    # tolerance. It is ~100x tighter than modulus's 4e-7 NAM parity gate.
    err = np.sqrt(np.mean((got - want) ** 2)) / (np.sqrt(np.mean(want**2)) + 1e-30)
    assert err < 1e-5, f"relative RMS {err:.3e}"


def test_state_interleaving_is_not_symmetric_so_the_test_could_have_caught_it(exported):
    """Guard on the guard.

    The readout indexes state as ``2 * block + part``. Swapping to
    ``part * n_blocks + block`` is the single easiest mistake to make in a Rust
    port and produces a plausible-looking model. Confirm the golden comparison
    actually discriminates it rather than passing by symmetry.
    """
    _, payload = exported
    x = np.asarray(payload["golden"]["input"], dtype=np.float64)
    want = np.asarray(payload["golden"]["output"], dtype=np.float64)

    stage = dict(payload["stages"][0])
    nb, out_ch = stage["n_blocks"], stage["out_ch"]
    cw = np.asarray(stage["c_w"], dtype=np.float64).reshape(out_ch, 2 * nb)
    blocked = np.empty_like(cw)
    blocked[:, :nb] = cw[:, 0::2]
    blocked[:, nb:] = cw[:, 1::2]
    stage["c_w"] = [float(v) for v in blocked.reshape(-1)]

    wrong = dict(payload)
    wrong["stages"] = [stage] + list(payload["stages"][1:])
    got = _run_cascade(wrong, x)

    err = np.sqrt(np.mean((got - want) ** 2)) / (np.sqrt(np.mean(want**2)) + 1e-30)
    assert err > 1e-2, f"wrong interleaving was not detected (rel RMS {err:.3e})"


def test_folded_coefficients_match_the_live_parameters(exported):
    """``a`` and ``b`` are precomputed at export; check the algebra, not just the
    shapes, so a folding bug cannot hide behind a self-consistent reader."""
    model, payload = exported
    stage = payload["stages"][0]
    bank = model.stage1

    with torch.no_grad():
        r = bank._radius()
        expected_a00 = (r * torch.cos(bank.theta)).numpy()
        expected_b = (bank.b_proj * torch.sqrt(1 - r * r)[:, None, None]).numpy()

    np.testing.assert_allclose(
        np.asarray(stage["a00"], dtype=np.float32), expected_a00, rtol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(stage["b"], dtype=np.float32).reshape(expected_b.shape),
        expected_b,
        rtol=1e-6,
    )
    # And the radii really are inside the unit circle, in the exported numbers.
    mags = np.hypot(
        np.asarray(stage["a00"], dtype=np.float64), np.asarray(stage["a10"], dtype=np.float64)
    )
    assert mags.max() < 1.0


def test_every_stage_declares_shapes_a_reader_can_size_buffers_from(exported):
    """A real-time backend allocates once in ``prepare``; it can only do that if the
    file states every dimension rather than implying it from array lengths."""
    _, payload = exported
    assert payload["schema"] == 1
    for stage in payload["stages"]:
        if stage["kind"] == "bank":
            nb, i, o = stage["n_blocks"], stage["in_ch"], stage["out_ch"]
            assert len(stage["b"]) == nb * 2 * i
            assert len(stage["c_w"]) == o * 2 * nb
            assert len(stage["c_b"]) == o
            for key in ("a00", "a01", "a10", "a11"):
                assert len(stage[key]) == nb
        else:
            ch, hid = stage["channels"], stage["hidden"]
            assert len(stage["w1"]) == hid * ch and len(stage["b1"]) == hid
            assert len(stage["w2"]) == ch * hid and len(stage["b2"]) == ch

    # Channel widths line up end to end: 1 in, 1 out.
    widths = []
    for stage in payload["stages"]:
        widths.append(
            (stage["in_ch"], stage["out_ch"])
            if stage["kind"] == "bank"
            else (stage["channels"], stage["channels"])
        )
    assert widths[0][0] == 1 and widths[-1][1] == 1
    for (_, out), (nxt, _) in zip(widths, widths[1:]):
        assert out == nxt


def test_golden_signal_actually_drives_the_nonlinearities(exported):
    """A probe that stays in the small-signal region would let a backend that got
    tanh wrong pass parity. Confirm the nonlinearities are meaningfully engaged."""
    model, payload = exported
    x = torch.tensor(payload["golden"]["input"], dtype=torch.float32).reshape(1, -1)
    with torch.no_grad():
        z = model.stage1(x.unsqueeze(1), streaming=True)
        pre = model.nl1.fc1(z.transpose(1, 2))
    # tanh departs from identity by >1% once |z| exceeds about 0.25.
    engaged = (pre.abs() > 0.25).float().mean()
    assert float(engaged) > 0.05, f"only {float(engaged):.1%} of tanh inputs are nonlinear"


def test_export_is_deterministic_for_a_fixed_seed():
    """Golden fixtures are checked into a repo and compared across machines."""
    torch.manual_seed(0)
    a = export_model(T.WienerHammersteinSSM().eval(), golden_samples=64)
    torch.manual_seed(0)
    b = export_model(T.WienerHammersteinSSM().eval(), golden_samples=64)
    assert a == b


def test_no_coefficient_is_nan_or_infinite(exported):
    """Model metadata crossing into a real-time thread is exactly where a poisoned
    scalar does damage; modulus's trait docs make the same point about loudness."""
    _, payload = exported
    for stage in payload["stages"]:
        for key, value in stage.items():
            if isinstance(value, list):
                arr = np.asarray(value, dtype=np.float64)
                assert np.isfinite(arr).all(), f"{stage['kind']}.{key}"
    assert all(math.isfinite(v) for v in payload["golden"]["output"])
