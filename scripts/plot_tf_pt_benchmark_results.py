#!/usr/bin/env python3
"""Build Figure: TF vs PT discrepancy summary (CPU vs GPU profiles) from benchmark JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]


def load_result(path: Path) -> dict:
    data = json.loads(path.read_text())
    if len(data.get("results", [])) != 1:
        raise ValueError(f"Expected exactly one profile in {path}")
    return data["results"][0]


def main() -> None:
    cpu_path = REPO / "results" / "tf_pt_benchmark_cpu.json"
    gpu_path = REPO / "results" / "tf_pt_benchmark.json"
    out_dir = REPO / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "tf_pt_benchmark_tf_pt_discrepancy.png"

    cpu = load_result(cpu_path)
    gpu = load_result(gpu_path)

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.0), constrained_layout=True)

    # Left: mean MAE (per window) for P and S — CPU vs GPU
    profiles = ("CPU\n(TF ref.)", "GPU\n(TF+PT CUDA)")
    x = np.arange(len(profiles))
    width = 0.35
    mae_p = [cpu["mae_p_mean"], gpu["mae_p_mean"]]
    mae_s = [cpu["mae_s_mean"], gpu["mae_s_mean"]]
    axes[0].bar(x - width / 2, mae_p, width, label="P branch", color="#2c7bb6")
    axes[0].bar(x + width / 2, mae_s, width, label="S branch", color="#d7191c")
    axes[0].set_ylabel(r"Mean MAE per window ($|\mathrm{TF}-\mathrm{PT}|$)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(profiles)
    axes[0].set_yscale("log")
    axes[0].set_title("Mean absolute error (probability traces)")
    axes[0].legend(frameon=True, fontsize=9)
    axes[0].grid(True, axis="y", which="both", alpha=0.3)

    # Right: mean of per-window max |TF−PT| (tighter is better)
    maxp = [cpu["per_window_max_abs_p_mean"], gpu["per_window_max_abs_p_mean"]]
    maxs = [cpu["per_window_max_abs_s_mean"], gpu["per_window_max_abs_s_mean"]]
    axes[1].bar(x - width / 2, maxp, width, label="P branch", color="#2c7bb6")
    axes[1].bar(x + width / 2, maxs, width, label="S branch", color="#d7191c")
    axes[1].set_ylabel(r"Mean of per-window $\max |\mathrm{TF}-\mathrm{PT}|$")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(profiles)
    axes[1].set_yscale("log")
    axes[1].set_title("Typical worst-point mismatch within each window")
    axes[1].legend(frameon=True, fontsize=9)
    axes[1].grid(True, axis="y", which="both", alpha=0.3)

    fig.suptitle(
        "TensorFlow versus PyTorch output agreement on SeisBench windows\n"
        r"(100k windows: TXED + STEAD, 50k per dataset, stride 1)",
        fontsize=11,
        y=1.02,
    )

    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    print("Wrote", out_png)


if __name__ == "__main__":
    main()
