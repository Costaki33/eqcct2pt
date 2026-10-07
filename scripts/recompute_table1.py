#!/usr/bin/env python3
"""Assemble Table 1 from one CPU JSON and one GPU JSON/NPZ of the same metric definitions.

Asserts MSE <= Dmax * MAE and MSE <= Dmax**2.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]


def _branch(rec: dict, prefix: str) -> dict:
    mae = float(rec[f"mae_{prefix}_mean"])
    mse = float(rec[f"mse_{prefix}_mean"])
    dmean = float(rec[f"per_window_max_abs_{prefix}_mean"])
    dmax = float(rec[f"global_max_abs_{prefix}"])
    frac = rec.get(f"fraction_windows_max_abs_{prefix}_below", {})
    share_1e4 = float(frac.get("0.0001", frac.get("1e-4", 1.0)))
    share_1e3 = float(frac.get("0.001", frac.get("1e-3", 1.0)))
    return _check(
        {
            "mae_mean": mae,
            "mse_mean": mse,
            "d_mean": dmean,
            "dmax": dmax,
            "share_le_1e-4": share_1e4,
            "share_le_1e-3": share_1e3,
        },
        prefix,
    )


def _from_npz(npz_path: Path, prefix: str) -> dict:
    z = np.load(npz_path)
    rec: dict = {"n_windows": int(z[f"{prefix}_p_mae"].size), "n_errors": 0}
    for br in ("p", "s"):
        mae = z[f"{prefix}_{br}_mae"].astype(np.float64)
        mse = z[f"{prefix}_{br}_mse"].astype(np.float64)
        d = z[f"{prefix}_{br}_max"].astype(np.float64)
        rec[br] = _check(
            {
                "mae_mean": float(mae.mean()),
                "mse_mean": float(mse.mean()),
                "d_mean": float(d.mean()),
                "dmax": float(d.max()),
                "share_le_1e-4": float((d <= 1e-4).mean()),
                "share_le_1e-3": float((d <= 1e-3).mean()),
                "n": int(mae.size),
            },
            f"{prefix} {br}",
        )
    return rec


def _check(row: dict, label: str) -> dict:
    bound = row["dmax"] * row["mae_mean"]
    bound2 = row["dmax"] ** 2
    mse = row["mse_mean"]
    if mse > bound * 1.01 + 1e-30:
        raise SystemExit(f"{label}: MSE={mse:.6e} exceeds Dmax*MAE={bound:.6e}")
    if mse > bound2 * 1.01 + 1e-30:
        raise SystemExit(f"{label}: MSE={mse:.6e} exceeds Dmax^2={bound2:.6e}")
    row["bound_D_times_MAE"] = bound
    row["bound_D2"] = bound2
    row["mse_ok"] = True
    return row


def _fmt(x: float) -> str:
    return f"{x:.2e}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu-json", type=Path, default=REPO / "results" / "tf_pt_benchmark_cpu.json")
    ap.add_argument("--cpu-npz", type=Path, default=None)
    ap.add_argument(
        "--gpu-json",
        type=Path,
        default=REPO / "results" / "tf_pt_benchmark_gpu_tf32off.json",
    )
    ap.add_argument("--gpu-npz", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=REPO / "results" / "table1_same_run.json")
    args = ap.parse_args()

    if args.cpu_npz and args.cpu_npz.exists():
        cpu_block = _from_npz(args.cpu_npz, "cpu")
        cpu_source = str(args.cpu_npz)
    else:
        cpu_raw = json.loads(args.cpu_json.read_text())["results"][0]
        cpu_block = {
            "n_windows": cpu_raw["n_windows"],
            "n_errors": cpu_raw.get("n_errors", 0),
            "p": _branch(cpu_raw, "p"),
            "s": _branch(cpu_raw, "s"),
        }
        cpu_source = str(args.cpu_json)
    if args.gpu_npz and args.gpu_npz.exists():
        gpu = _from_npz(args.gpu_npz, "gpu0")
        gpu_source = str(args.gpu_npz)
    else:
        gpu_raw = json.loads(args.gpu_json.read_text())["results"][0]
        gpu = {
            "n_windows": gpu_raw["n_windows"],
            "n_errors": gpu_raw.get("n_errors", 0),
            "p": _branch(gpu_raw, "p"),
            "s": _branch(gpu_raw, "s"),
        }
        gpu_source = str(args.gpu_json)
    payload = {
        "cpu_source": cpu_source,
        "gpu_source": gpu_source,
        "cpu": cpu_block,
        "gpu": gpu,
        "aggregation": (
            "Mean MAE/MSE are averages of per-window MAE_w/MSE_w over N windows; "
            "each window averages T=6000 samples. D_w is the per-window max |TF-PT|; "
            "Dmax is the dataset-wide max."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print("Table 1 (scientific notation):")
    for dev in ("cpu", "gpu"):
        for br in ("p", "s"):
            d = payload[dev][br]
            print(
                f"  {dev.upper()} {br.upper()}: MAE={_fmt(d['mae_mean'])} "
                f"MSE={_fmt(d['mse_mean'])} meanDw={_fmt(d['d_mean'])} "
                f"Dmax={_fmt(d['dmax'])}"
            )


if __name__ == "__main__":
    main()
