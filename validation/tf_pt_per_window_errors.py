#!/usr/bin/env python3
"""
Collect per-window TF vs PT discrepancy on a (small) SeisBench subset, for
distribution plots (CDFs / violins / boxplots).

This is the lightweight cousin of ``tf_pt_seisbench_dataset_benchmark`` — it
runs over far fewer windows (default 1000 per dataset per profile) but stores
the raw per-window arrays needed for distribution plots:

  * MAE per window for the P branch and the S branch
  * max |TF - PT| per window for the P branch and the S branch (exported as ``*_max``)
  * median |TF - PT| within each window (over output samples), P and S branches (``*_med``)

Each device profile (``cpu``, ``gpu0``, ``gpu1``) runs in its own subprocess
(TF CPU mode hides GPUs globally and would otherwise break later GPU profiles).

Output is a single NPZ keyed by ``<profile>_<branch>_<metric>``, e.g.
``cpu_p_mae``, ``gpu0_s_max``, ``gpu0_p_med``.

Example::

  PYTHONPATH=. python -m validation.tf_pt_per_window_errors \\
    --profiles cpu,gpu0 --max-windows 1000 \\
    --output-npz results/per_window_errors.npz
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from paths import MODELPS_DIR, REPO_ROOT


def parse_profiles(s: str, n_cuda: int) -> list[tuple[str, str, str]]:
    out: list[tuple[str, str, str]] = []
    for part in s.split(","):
        part = part.strip().lower()
        if part == "cpu":
            out.append(("cpu", "/CPU:0", "cpu"))
        elif part in ("gpu0", "cuda0", "0"):
            if n_cuda < 1:
                continue
            out.append(("gpu0", "/GPU:0", "cuda:0"))
        elif part in ("gpu1", "cuda1", "1"):
            if n_cuda < 2:
                continue
            out.append(("gpu1", "/GPU:1", "cuda:1"))
        else:
            raise ValueError(part)
    return out


def norm_std_time(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)


def build_window_from_idx(dataset, idx: int, p_sample: int, s_sample: int) -> Optional[np.ndarray]:
    raw = np.asarray(dataset.get_waveforms([idx])[0], dtype=np.float32)
    if raw.shape[1] < 6000:
        return None
    start = max(0, min(p_sample - 3000, raw.shape[1] - 6000))
    s_in = s_sample - start
    if not (0 <= s_in < 6000):
        return None
    win = raw[:, start : start + 6000]
    return norm_std_time(win.transpose(1, 0)[np.newaxis, ...])


def run_profile(
    profile: str,
    tf_device: str,
    torch_device: str,
    *,
    p_h5: Path,
    s_h5: Path,
    pt_p: Optional[Path],
    pt_s: Optional[Path],
    datasets: list[str],
    max_windows: int,
    stride: int,
) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import torch

    # GPU manuscript runs: disable TF32 (matmul + cudnn + TF) unless overridden.
    if not tf_device.startswith("/CPU") and os.environ.get("EQCCT_ALLOW_TF32", "").lower() not in (
        "1",
        "true",
        "yes",
    ):
        from validation.precision import configure_tf32

        configure_tf32(disable=True)

    if tf_device.startswith("/CPU"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    from reference.predictor_tf import load_eqcct_model
    from models.eqcct import EQCCTModelP, EQCCTModelS
    from conversion.loader import load_eqcct_model_p_weights, load_eqcct_model_s_weights

    model_p_tf, model_s_tf = load_eqcct_model(str(p_h5), str(s_h5))
    m_p = EQCCTModelP().eval()
    m_s = EQCCTModelS().eval()
    if pt_p and pt_p.is_file():
        ck = torch.load(pt_p, map_location="cpu", weights_only=False)
        m_p.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_p_weights(m_p, h5_path=str(p_h5))
    if pt_s and pt_s.is_file():
        ck = torch.load(pt_s, map_location="cpu", weights_only=False)
        m_s.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_s_weights(m_s, h5_path=str(s_h5))
    m_p = m_p.to(torch_device)
    m_s = m_s.to(torch_device)

    try:
        import seisbench.data as sbd
    except ImportError as e:
        raise SystemExit("seisbench is required: pip install seisbench") from e
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x

    mae_p_list: list[float] = []
    mae_s_list: list[float] = []
    mse_p_list: list[float] = []
    mse_s_list: list[float] = []
    max_p_list: list[float] = []
    max_s_list: list[float] = []
    med_p_list: list[float] = []
    med_s_list: list[float] = []
    src_dataset: list[str] = []
    batch_size = int(os.environ.get("EQCCT_PER_WINDOW_BATCH", "8"))

    def flush(wfs: list[np.ndarray], ds_labels: list[str]) -> None:
        if not wfs:
            return
        batch = np.concatenate(wfs, axis=0)  # (B, T, 3)
        p_tf_t = model_p_tf(tf.constant(batch, dtype=tf.float32), training=False)
        s_tf_t = model_s_tf(tf.constant(batch, dtype=tf.float32), training=False)
        p_tf = p_tf_t.numpy() if hasattr(p_tf_t, "numpy") else np.asarray(p_tf_t)
        s_tf = s_tf_t.numpy() if hasattr(s_tf_t, "numpy") else np.asarray(s_tf_t)
        if p_tf.ndim == 3:
            p_tf = p_tf[..., 0]
        if s_tf.ndim == 3:
            s_tf = s_tf[..., 0]
        with torch.no_grad():
            x_pt = torch.from_numpy(batch).to(torch_device)
            p_pt = m_p(x_pt).detach().cpu().numpy()
            s_pt = m_s(x_pt).detach().cpu().numpy()
            if p_pt.ndim == 3:
                p_pt = p_pt[..., 0]
            if s_pt.ndim == 3:
                s_pt = s_pt[..., 0]
        for i, ds_lab in enumerate(ds_labels):
            d_p = np.abs(p_tf[i] - p_pt[i]).ravel()
            d_s = np.abs(s_tf[i] - s_pt[i]).ravel()
            mae_p_list.append(float(d_p.mean()))
            mae_s_list.append(float(d_s.mean()))
            mse_p_list.append(float(np.mean(d_p**2)))
            mse_s_list.append(float(np.mean(d_s**2)))
            max_p_list.append(float(d_p.max()))
            max_s_list.append(float(d_s.max()))
            med_p_list.append(float(np.median(d_p)))
            med_s_list.append(float(np.median(d_s)))
            src_dataset.append(ds_lab)
        n_done = len(mae_p_list)
        if n_done and n_done % 2000 == 0:
            print(
                f"[mae] n={n_done} P={float(np.mean(mae_p_list)):.4e} "
                f"S={float(np.mean(mae_s_list)):.4e} "
                f"(Table 1 CPU: 1.38e-9 / 1.36e-9)",
                flush=True,
            )

    for ds_name in datasets:
        if ds_name.lower() == "txed":
            ds = sbd.TXED(sampling_rate=100, component_order="ZNE")
        elif ds_name.lower() == "stead":
            ds = sbd.STEAD(sampling_rate=100, component_order="ZNE")
        else:
            raise ValueError(ds_name)
        md = ds.metadata
        good_p = md["trace_p_arrival_sample"].notna() & (md["trace_p_arrival_sample"] > 0)
        good_s = md["trace_s_arrival_sample"].notna() & (md["trace_s_arrival_sample"] > 0)
        sub = md[good_p & good_s]
        indices = list(range(0, len(sub), max(1, stride)))
        if max_windows:
            indices = indices[:max_windows]

        buf_wf: list[np.ndarray] = []
        buf_ds: list[str] = []
        for j in tqdm(indices, desc=f"{profile}/{ds_name}", file=sys.stdout):
            row = sub.iloc[j]
            try:
                idx = ds.get_idx_from_trace_name(row["trace_name"])
                wf = build_window_from_idx(ds, idx, int(row["trace_p_arrival_sample"]),
                                           int(row["trace_s_arrival_sample"]))
            except Exception:
                continue
            if wf is None:
                continue
            buf_wf.append(wf)
            buf_ds.append(ds_name)
            if len(buf_wf) >= batch_size:
                try:
                    flush(buf_wf, buf_ds)
                except Exception as exc:
                    print(f"[warn] batch flush failed ({len(buf_wf)} windows): {exc!r}", flush=True)
                    # Fall back to single-window inference so a bad batch does not drop data.
                    for wf_i, ds_i in zip(buf_wf, buf_ds):
                        try:
                            flush([wf_i], [ds_i])
                        except Exception as exc2:
                            print(f"[warn] single-window flush failed: {exc2!r}", flush=True)
                buf_wf, buf_ds = [], []
        try:
            flush(buf_wf, buf_ds)
        except Exception as exc:
            print(f"[warn] final flush failed ({len(buf_wf)} windows): {exc!r}", flush=True)
            for wf_i, ds_i in zip(buf_wf, buf_ds):
                try:
                    flush([wf_i], [ds_i])
                except Exception as exc2:
                    print(f"[warn] single-window flush failed: {exc2!r}", flush=True)

    return {
        "profile": profile,
        "n_windows": len(mae_p_list),
        "mae_p": np.asarray(mae_p_list, dtype=np.float64),
        "mae_s": np.asarray(mae_s_list, dtype=np.float64),
        "mse_p": np.asarray(mse_p_list, dtype=np.float64),
        "mse_s": np.asarray(mse_s_list, dtype=np.float64),
        "max_p": np.asarray(max_p_list, dtype=np.float64),
        "max_s": np.asarray(max_s_list, dtype=np.float64),
        "med_p": np.asarray(med_p_list, dtype=np.float64),
        "med_s": np.asarray(med_s_list, dtype=np.float64),
        "dataset": np.asarray(src_dataset, dtype=object),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--pt-p", type=Path, default=None)
    parser.add_argument("--pt-s", type=Path, default=None)
    parser.add_argument("--datasets", type=str, default="txed,stead")
    parser.add_argument("--max-windows", type=int, default=1000, help="Per dataset")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--profiles", type=str, default="cpu,gpu0")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--output-summary-json", type=Path, default=None)
    args = parser.parse_args()

    repo = args.repo or REPO_ROOT
    p_h5 = args.p_h5 or MODELPS_DIR / "test_trainer_024.h5"
    s_h5 = args.s_h5 or MODELPS_DIR / "test_trainer_021.h5"
    pt_p = args.pt_p or (MODELPS_DIR / "eqcct_model_p.pt")
    pt_s = args.pt_s or (MODELPS_DIR / "eqcct_model_s.pt")
    if not pt_p.is_file():
        pt_p = None
    if not pt_s.is_file():
        pt_s = None
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

    ds_list = (["txed", "stead"] if args.datasets.lower() == "both"
               else [x.strip().lower() for x in args.datasets.split(",") if x.strip()])

    import torch
    n_cuda = torch.cuda.device_count()
    profiles = parse_profiles(args.profiles, n_cuda)
    if not profiles:
        raise SystemExit("No valid profiles.")

    arrays: dict[str, np.ndarray] = {}
    summary: list[dict] = []

    if len(profiles) > 1:
        child_env = os.environ.copy()
        child_env["EQCCT_TF_PT_QUIET_JSON"] = "1"
        repo_pp = str(repo)
        child_env["PYTHONPATH"] = (
            repo_pp + os.pathsep + child_env["PYTHONPATH"]
            if child_env.get("PYTHONPATH") else repo_pp
        )
        with tempfile.TemporaryDirectory() as tmpd:
            tmp = Path(tmpd)
            for name, _tfd, _ptd in profiles:
                child_npz = tmp / f"per_window_{name}.npz"
                cmd = [
                    sys.executable, "-m", "validation.tf_pt_per_window_errors",
                    "--repo", str(repo),
                    "--p-h5", str(p_h5), "--s-h5", str(s_h5),
                    "--datasets", args.datasets,
                    "--max-windows", str(args.max_windows),
                    "--stride", str(args.stride),
                    "--profiles", name,
                    "--output-npz", str(child_npz),
                ]
                if pt_p is not None:
                    cmd.extend(["--pt-p", str(pt_p)])
                if pt_s is not None:
                    cmd.extend(["--pt-s", str(pt_s)])
                print(f"\n========== Profile {name} (subprocess) ==========")
                print("[info]", " ".join(cmd))
                subprocess.check_call(cmd, env=child_env, cwd=str(repo))
                z = np.load(child_npz, allow_pickle=True)
                for k in z.files:
                    arrays[k] = z[k]
                summary.append({"profile": name, "n_windows": int(z[f"{name}_p_mae"].size)})
    else:
        name, tf_dev, pt_dev = profiles[0]
        d = run_profile(
            name, tf_dev, pt_dev,
            p_h5=p_h5, s_h5=s_h5, pt_p=pt_p, pt_s=pt_s,
            datasets=ds_list, max_windows=args.max_windows, stride=args.stride,
        )
        for branch_short, key in (
            ("p_mae", "mae_p"),
            ("s_mae", "mae_s"),
            ("p_mse", "mse_p"),
            ("s_mse", "mse_s"),
            ("p_max", "max_p"),
            ("s_max", "max_s"),
            ("p_med", "med_p"),
            ("s_med", "med_s"),
            ("dataset", "dataset"),
        ):
            arrays[f"{name}_{branch_short}"] = d[key]
        summary.append({"profile": name, "n_windows": d["n_windows"]})

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **arrays)
    print("[info] Wrote", args.output_npz)

    if args.output_summary_json is not None:
        args.output_summary_json.write_text(json.dumps(
            {"profiles": [p[0] for p in profiles], "summary": summary,
             "datasets": ds_list, "max_windows": args.max_windows, "stride": args.stride},
            indent=2))


if __name__ == "__main__":
    main()
