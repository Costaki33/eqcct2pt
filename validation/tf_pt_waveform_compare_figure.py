#!/usr/bin/env python3
"""
Waveform comparison figure: TF vs PT P/S probability traces on six real
SeisBench windows (default: 3 from TXED + 3 from STEAD).

For each chosen trace we plot

  row 1 — three-channel ZNE waveform with catalog P/S markers
  row 2 — P-branch probability (TF and PT overlaid, with catalog P marker)
  row 3 — S-branch probability (TF and PT overlaid, with catalog S marker)

Each trace becomes one column in the final figure (3 traces × 2 datasets = 6
columns). TensorFlow runs on CPU here; CPU/GPU outputs differ at the
``< 10^{-3}`` level (see the dataset benchmark) which is invisible at this
scale, so a single device profile is sufficient for visual parity.

Example::

  PYTHONPATH=. python -m validation.tf_pt_waveform_compare_figure \\
    --n-per-dataset 3 --seed 0 --output figures/tf_pt_waveform_overlays.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

from paths import MODELPS_DIR, REPO_ROOT


def norm_std_time(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)


def pick_windows(dataset, names: list[str], metadata, n_target: int, seed: int):
    """Return up to ``n_target`` qualifying ``(trace_name, wf, p_in, s_in, start)`` tuples."""
    import random

    out = []
    rng = random.Random(seed)
    shuffled = list(names)
    rng.shuffle(shuffled)
    for trace_name in shuffled:
        if len(out) >= n_target:
            break
        try:
            idx = dataset.get_idx_from_trace_name(trace_name)
        except Exception:
            continue
        try:
            raw = np.asarray(dataset.get_waveforms([idx])[0], dtype=np.float32)
        except Exception:
            continue
        if raw.shape[1] < 6000:
            continue
        p_sample = int(metadata.loc[idx, "trace_p_arrival_sample"])
        s_sample = int(metadata.loc[idx, "trace_s_arrival_sample"])
        start = max(0, min(p_sample - 3000, raw.shape[1] - 6000))
        p_in = p_sample - start
        s_in = s_sample - start
        if not (0 <= s_in < 6000):
            continue
        win = raw[:, start : start + 6000]
        wf = norm_std_time(win.transpose(1, 0)[np.newaxis, ...])
        out.append((trace_name, wf, p_in, s_in, start))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--pt-p", type=Path, default=None)
    parser.add_argument("--pt-s", type=Path, default=None)
    parser.add_argument("--datasets", type=str, default="txed,stead", help="Comma list")
    parser.add_argument("--n-per-dataset", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo or REPO_ROOT
    p_h5 = args.p_h5 or MODELPS_DIR / "test_trainer_024.h5"
    s_h5 = args.s_h5 or MODELPS_DIR / "test_trainer_021.h5"
    pt_p = args.pt_p or (MODELPS_DIR / "eqcct_model_p.pt")
    pt_s = args.pt_s or (MODELPS_DIR / "eqcct_model_s.pt")
    out_path = args.output or (repo / "figures" / "tf_pt_waveform_overlays.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import matplotlib.pyplot as plt
    import tensorflow as tf
    import torch
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass

    from reference.predictor_tf import load_eqcct_model
    from models.predictor_pt_p import EQCCTModelP, EQCCTModelS
    from conversion.loader import load_eqcct_model_p_weights, load_eqcct_model_s_weights

    model_p_tf, model_s_tf = load_eqcct_model(str(p_h5), str(s_h5))
    m_p = EQCCTModelP().eval()
    m_s = EQCCTModelS().eval()
    if pt_p.is_file():
        ck = torch.load(pt_p, map_location="cpu", weights_only=False)
        m_p.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_p_weights(m_p, h5_path=str(p_h5))
    if pt_s.is_file():
        ck = torch.load(pt_s, map_location="cpu", weights_only=False)
        m_s.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_s_weights(m_s, h5_path=str(s_h5))

    try:
        import seisbench.data as sbd
    except ImportError as e:
        raise SystemExit("seisbench is required: pip install seisbench") from e

    ds_names = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    sample_rate = 100

    selected: list[tuple[str, str, np.ndarray, int, int]] = []
    for ds_label in ds_names:
        if ds_label == "txed":
            ds = sbd.TXED(sampling_rate=sample_rate, component_order="ZNE")
        elif ds_label == "stead":
            ds = sbd.STEAD(sampling_rate=sample_rate, component_order="ZNE")
        else:
            raise ValueError(ds_label)
        md = ds.metadata
        good_p = md["trace_p_arrival_sample"].notna() & (md["trace_p_arrival_sample"] > 0)
        good_s = md["trace_s_arrival_sample"].notna() & (md["trace_s_arrival_sample"] > 0)
        names = md[good_p & good_s]["trace_name"].tolist()
        picks = pick_windows(ds, names, md, args.n_per_dataset, args.seed)
        if len(picks) < args.n_per_dataset:
            print(f"[warn] {ds_label}: only {len(picks)} qualifying traces found")
        for trace_name, wf, p_in, s_in, _start in picks:
            selected.append((ds_label.upper(), trace_name, wf, p_in, s_in))

    if not selected:
        raise SystemExit("No traces selected.")

    # Stack traces vertically in blocks of 3 (e.g. 3 TXED on top, 3 STEAD below).
    n_traces = len(selected)
    block_cols = 3
    n_blocks = (n_traces + block_cols - 1) // block_cols
    n_rows_total = 3 * n_blocks
    fig_w = 3.5 * block_cols
    fig_h = 3.4 * n_blocks + 0.6
    fig, axes = plt.subplots(
        n_rows_total, block_cols, figsize=(fig_w, fig_h), sharex=True, constrained_layout=True,
    )
    if n_rows_total == 1:
        axes = axes.reshape(1, -1)
    elif block_cols == 1:
        axes = axes.reshape(-1, 1)

    for blk in range(n_blocks):
        for slot in range(block_cols):
            i = blk * block_cols + slot
            row_w = 3 * blk
            row_p = row_w + 1
            row_s = row_w + 2
            if i >= n_traces:
                for r in (row_w, row_p, row_s):
                    axes[r, slot].axis("off")
                continue
            ds_label, trace_name, wf, p_in, s_in = selected[i]

            out_p_tf = model_p_tf(wf, training=False)
            p_tf = out_p_tf.numpy() if hasattr(out_p_tf, "numpy") else np.asarray(out_p_tf)
            if p_tf.ndim == 3:
                p_tf = p_tf[..., 0]
            out_s_tf = model_s_tf(wf, training=False)
            s_tf = out_s_tf.numpy() if hasattr(out_s_tf, "numpy") else np.asarray(out_s_tf)
            if s_tf.ndim == 3:
                s_tf = s_tf[..., 0]
            with torch.no_grad():
                p_pt = m_p(torch.from_numpy(wf)).numpy()[..., 0]
                s_pt = m_s(torch.from_numpy(wf)).numpy()[..., 0]

            d_p = np.abs(p_tf - p_pt).max()
            d_s = np.abs(s_tf - s_pt).max()

            ax_w = axes[row_w, slot]
            offsets = (-3.0, 0.0, 3.0)
            labels = ("Z", "N", "E")
            for k, (off, lab) in enumerate(zip(offsets, labels)):
                ax_w.plot(wf[0, :, k] + off, lw=0.5, color="0.15")
                ax_w.text(-180, off, lab, va="center", ha="right", fontsize=8)
            ax_w.axvline(p_in, color="C0", ls="--", lw=0.9, alpha=0.85)
            ax_w.axvline(s_in, color="C1", ls="--", lw=0.9, alpha=0.85)
            ax_w.set_title(f"{ds_label}  {trace_name[:28]}…", fontsize=9)
            ax_w.set_yticks([])
            ax_w.set_ylim(-6.0, 6.0)
            ax_w.grid(True, axis="x", alpha=0.25)

            ax_p = axes[row_p, slot]
            ax_p.plot(p_tf[0], color="#1f77b4", lw=1.0, label="TF P", alpha=0.9)
            ax_p.plot(p_pt[0], color="#d62728", lw=1.0, ls="--", label="PT P", alpha=0.85)
            ax_p.axvline(p_in, color="C0", ls=":", lw=0.7, alpha=0.7)
            ax_p.set_ylim(-0.05, 1.05)
            if slot == 0:
                ax_p.set_ylabel("P prob", fontsize=8)
            ax_p.text(0.02, 0.92, f"max|TF-PT|={d_p:.2e}", transform=ax_p.transAxes, fontsize=7,
                      va="top", bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.18"))
            if slot == 0 and blk == 0:
                ax_p.legend(loc="upper right", fontsize=7)
            ax_p.grid(True, alpha=0.25)

            ax_s = axes[row_s, slot]
            ax_s.plot(s_tf[0], color="#1f77b4", lw=1.0, label="TF S", alpha=0.9)
            ax_s.plot(s_pt[0], color="#d62728", lw=1.0, ls="--", label="PT S", alpha=0.85)
            ax_s.axvline(s_in, color="C1", ls=":", lw=0.7, alpha=0.7)
            ax_s.set_ylim(-0.05, 1.05)
            if slot == 0:
                ax_s.set_ylabel("S prob", fontsize=8)
            if blk == n_blocks - 1:
                ax_s.set_xlabel("sample (in 6000-pt window)", fontsize=8)
            ax_s.text(0.02, 0.92, f"max|TF-PT|={d_s:.2e}", transform=ax_s.transAxes, fontsize=7,
                      va="top", bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85, boxstyle="round,pad=0.18"))
            if slot == 0 and blk == 0:
                ax_s.legend(loc="upper right", fontsize=7)
            ax_s.grid(True, alpha=0.25)

    fig.suptitle(
        "TensorFlow vs PyTorch probability traces on real SeisBench windows\n"
        "(dashed PT curves overlay solid TF curves; vertical dashes mark catalog P / S)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print("[info] Wrote", out_path)


if __name__ == "__main__":
    main()
