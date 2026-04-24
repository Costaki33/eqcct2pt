#!/usr/bin/env python3
"""
Scan SeisBench TXED and/or STEAD: compare EQCCT TensorFlow vs PyTorch (P and S) on
identical 6000-sample ZNE windows (P-centered; catalog S required inside window).

For each *device profile* (CPU, first GPU, second GPU, …), loads models on that
device, runs the same windows, and aggregates MSE / MAE / max-abs-diff and
fractions below tolerance thresholds.

If you pass **multiple** profiles (e.g. ``cpu,gpu0,gpu1``), each profile runs in a
**separate Python process** so TensorFlow CPU mode (which hides GPUs) does not
break later GPU profiles.

**Full datasets** can be millions of windows — use ``--max-windows`` for dry runs,
or ``--stride`` to subsample (e.g. every 10th qualifying trace).

Example (quick smoke test, CPU only), from the **methods bundle root**::

  cd /path/to/methods_tf_to_pt_contribution
  PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_seisbench_dataset_benchmark \\
    --datasets txed --max-windows 200 --profiles cpu

Example (TXED + STEAD, all three profiles, cap 50k windows per dataset)::

  PYTHONPATH=src python -m eqcct_tf_pt_transfer.validation.tf_pt_seisbench_dataset_benchmark \\
    --datasets both --max-windows 50000 --profiles cpu,gpu0,gpu1 \\
    --output-json results/tf_pt_benchmark.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Online stats (Welford) for mean / variance of scalar stream
# ---------------------------------------------------------------------------


class Welford:
    def __init__(self) -> None:
        self.n = 0
        self._mean = 0.0
        self._m2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        d = x - self._mean
        self._mean += d / self.n
        d2 = x - self._mean
        self._m2 += d * d2

    @property
    def mean(self) -> float:
        return self._mean if self.n else float("nan")

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return math.sqrt(self._m2 / (self.n - 1))


@dataclass
class ThresholdCounter:
    """Count windows where max |TF-PT| is below each threshold."""

    thresholds: tuple[float, ...] = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
    counts: dict[float, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for t in self.thresholds:
            self.counts.setdefault(t, 0)

    def observe_max_abs(self, max_abs: float) -> None:
        for t in self.thresholds:
            if max_abs <= t:
                self.counts[t] += 1


@dataclass
class ProfileStats:
    profile: str
    n_windows: int = 0
    n_skipped_short: int = 0
    n_skipped_s_outside: int = 0
    n_errors: int = 0
    mse_p: Welford = field(default_factory=Welford)
    mse_s: Welford = field(default_factory=Welford)
    mae_p: Welford = field(default_factory=Welford)
    mae_s: Welford = field(default_factory=Welford)
    max_abs_p: Welford = field(default_factory=Welford)  # distribution of per-window max |diff|
    max_abs_s: Welford = field(default_factory=Welford)
    global_max_p: float = 0.0
    global_max_s: float = 0.0
    thr_p: ThresholdCounter = field(default_factory=ThresholdCounter)
    thr_s: ThresholdCounter = field(default_factory=ThresholdCounter)
    seconds_elapsed: float = 0.0

    def to_dict(self) -> dict:
        return {
            "profile": self.profile,
            "n_windows": self.n_windows,
            "n_skipped_short": self.n_skipped_short,
            "n_skipped_s_outside": self.n_skipped_s_outside,
            "n_errors": self.n_errors,
            "mse_p_mean": self.mse_p.mean,
            "mse_p_std": self.mse_p.std,
            "mse_s_mean": self.mse_s.mean,
            "mse_s_std": self.mse_s.std,
            "mae_p_mean": self.mae_p.mean,
            "mae_p_std": self.mae_p.std,
            "mae_s_mean": self.mae_s.mean,
            "mae_s_std": self.mae_s.std,
            "per_window_max_abs_p_mean": self.max_abs_p.mean,
            "per_window_max_abs_p_std": self.max_abs_p.std,
            "per_window_max_abs_s_mean": self.max_abs_s.mean,
            "per_window_max_abs_s_std": self.max_abs_s.std,
            "global_max_abs_p": self.global_max_p,
            "global_max_abs_s": self.global_max_s,
            "fraction_windows_max_abs_p_below": {
                str(t): self.thr_p.counts[t] / self.n_windows if self.n_windows else 0.0
                for t in self.thr_p.thresholds
            },
            "fraction_windows_max_abs_s_below": {
                str(t): self.thr_s.counts[t] / self.n_windows if self.n_windows else 0.0
                for t in self.thr_s.thresholds
            },
            "seconds_elapsed": self.seconds_elapsed,
        }


def norm_std_time(x: np.ndarray) -> np.ndarray:
    """(B, T, C): demean and scale each channel over time."""
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)


def tf_forward_np(model, wf_np: np.ndarray, tf_device: str) -> np.ndarray:
    import tensorflow as tf

    x = tf.constant(wf_np, dtype=tf.float32)
    with tf.device(tf_device):
        out = model(x, training=False)
    if hasattr(out, "numpy"):
        out = out.numpy()
    else:
        out = np.asarray(out)
    if out.ndim == 3:
        out = out[..., 0]
    return np.asarray(out, dtype=np.float32)


def pt_forward_np(model, wf_np: np.ndarray, torch_device: str) -> np.ndarray:
    import torch

    dev = torch.device(torch_device)
    t = torch.from_numpy(wf_np).to(dev)
    model_dev = model.to(dev)
    model_dev.eval()
    with torch.no_grad():
        out = model_dev(t)
    out = out.detach().cpu().numpy()
    if out.ndim == 3:
        out = out[..., 0]
    return np.asarray(out, dtype=np.float32)


def window_metrics(
    p_tf: np.ndarray, p_pt: np.ndarray, s_tf: np.ndarray, s_pt: np.ndarray
) -> tuple[float, float, float, float, float, float]:
    d_p = p_tf.ravel() - p_pt.ravel()
    d_s = s_tf.ravel() - s_pt.ravel()
    mse_p = float(np.mean(d_p**2))
    mse_s = float(np.mean(d_s**2))
    mae_p = float(np.mean(np.abs(d_p)))
    mae_s = float(np.mean(np.abs(d_s)))
    max_p = float(np.max(np.abs(d_p)))
    max_s = float(np.max(np.abs(d_s)))
    return mse_p, mse_s, mae_p, mae_s, max_p, max_s


def build_window_from_idx(dataset, idx: int, p_sample: int, s_sample: int) -> Optional[np.ndarray]:
    """Return (1,6000,3) float32 window or None if infeasible."""
    raw = np.asarray(dataset.get_waveforms([idx])[0], dtype=np.float32)
    if raw.shape[1] < 6000:
        return None
    start = max(0, min(p_sample - 3000, raw.shape[1] - 6000))
    s_in = s_sample - start
    if not (0 <= s_in < 6000):
        return None
    win = raw[:, start : start + 6000]
    return norm_std_time(win.transpose(1, 0)[np.newaxis, ...])


def parse_profiles(
    s: str,
    n_cuda: int,
) -> list[tuple[str, str, str]]:
    """
    Returns list of (name, tf_device, torch_device).
    - cpu -> /CPU:0, cpu
    - gpu0 -> /GPU:0, cuda:0
    - gpu1 -> /GPU:1, cuda:1 (if available)
    """
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
            raise ValueError(f"Unknown profile token: {part}")
    if not out:
        raise ValueError("No valid profiles (check GPU availability).")
    return out


def run_profile(
    profile_name: str,
    tf_device: str,
    torch_device: str,
    *,
    datasets: list[str],
    repo: Path,
    p_h5: Path,
    s_h5: Path,
    pt_p: Optional[Path],
    pt_s: Optional[Path],
    max_windows: int,
    stride: int,
    tqdm_disable: bool,
) -> ProfileStats:
    import tensorflow as tf
    import torch

    stats = ProfileStats(profile=profile_name)
    t0 = time.perf_counter()

    # TensorFlow: optional hide GPUs for CPU profile
    if tf_device.startswith("/CPU"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    else:
        # Ensure at least one GPU visible for GPU profiles
        try:
            gpus = tf.config.list_physical_devices("GPU")
            if not gpus:
                raise RuntimeError(
                    f"Profile {profile_name} needs GPU but TensorFlow lists no GPUs. "
                    "Use --profiles cpu or fix TF GPU install."
                )
        except Exception:
            pass

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from eqcct_tf_pt_transfer.reference.predictor_tf import load_eqcct_model
    from eqcct_tf_pt_transfer.models.predictor_pt_p import EQCCTModelP, EQCCTModelS
    from eqcct_tf_pt_transfer.conversion.loader import load_eqcct_model_p_weights, load_eqcct_model_s_weights

    model_p_tf, model_s_tf = load_eqcct_model(str(p_h5), str(s_h5))

    model_p_pt = EQCCTModelP().eval()
    model_s_pt = EQCCTModelS().eval()
    if pt_p and pt_p.is_file():
        ck = torch.load(pt_p, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            model_p_pt.load_state_dict(ck["state_dict"])
        else:
            model_p_pt.load_state_dict(ck)
    else:
        load_eqcct_model_p_weights(model_p_pt, h5_path=str(p_h5))
    if pt_s and pt_s.is_file():
        ck = torch.load(pt_s, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "state_dict" in ck:
            model_s_pt.load_state_dict(ck["state_dict"])
        else:
            model_s_pt.load_state_dict(ck)
    else:
        load_eqcct_model_s_weights(model_s_pt, h5_path=str(s_h5))

    try:
        import seisbench.data as sbd
    except ImportError as e:
        raise SystemExit("seisbench is required: pip install seisbench") from e

    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x

    sample_rate = 100

    for ds_name in datasets:
        if ds_name.lower() == "txed":
            ds = sbd.TXED(sampling_rate=sample_rate, component_order="ZNE")
        elif ds_name.lower() == "stead":
            ds = sbd.STEAD(sampling_rate=sample_rate, component_order="ZNE")
        else:
            raise ValueError(ds_name)

        md = ds.metadata
        # Build iterator: qualifying rows with stride/max_windows
        good_p = md["trace_p_arrival_sample"].notna() & (md["trace_p_arrival_sample"] > 0)
        good_s = md["trace_s_arrival_sample"].notna() & (md["trace_s_arrival_sample"] > 0)
        sub = md[good_p & good_s]
        indices = list(range(0, len(sub), max(1, stride)))
        if max_windows:
            indices = indices[:max_windows]

        it = indices
        if not tqdm_disable:
            it = tqdm(
                it,
                desc=f"{profile_name}/{ds_name}",
                unit="win",
                file=sys.stdout,
            )

        for j in it:
            row = sub.iloc[j]
            trace_name = row["trace_name"]
            try:
                idx = ds.get_idx_from_trace_name(trace_name)
            except Exception:
                stats.n_errors += 1
                continue
            p_sample = int(row["trace_p_arrival_sample"])
            s_sample = int(row["trace_s_arrival_sample"])
            try:
                wf = build_window_from_idx(ds, idx, p_sample, s_sample)
            except Exception:
                stats.n_errors += 1
                continue
            if wf is None:
                raw = np.asarray(ds.get_waveforms([idx])[0], dtype=np.float32)
                if raw.shape[1] < 6000:
                    stats.n_skipped_short += 1
                else:
                    stats.n_skipped_s_outside += 1
                continue

            try:
                p_tf = tf_forward_np(model_p_tf, wf, tf_device)
                s_tf = tf_forward_np(model_s_tf, wf, tf_device)
                p_pt = pt_forward_np(model_p_pt, wf, torch_device)
                s_pt = pt_forward_np(model_s_pt, wf, torch_device)
            except Exception:
                stats.n_errors += 1
                continue

            mse_p, mse_s, mae_p, mae_s, max_p, max_s = window_metrics(
                p_tf, p_pt, s_tf, s_pt
            )
            stats.mse_p.update(mse_p)
            stats.mse_s.update(mse_s)
            stats.mae_p.update(mae_p)
            stats.mae_s.update(mae_s)
            stats.max_abs_p.update(max_p)
            stats.max_abs_s.update(max_s)
            stats.global_max_p = max(stats.global_max_p, max_p)
            stats.global_max_s = max(stats.global_max_s, max_s)
            stats.thr_p.observe_max_abs(max_p)
            stats.thr_s.observe_max_abs(max_s)
            stats.n_windows += 1

    stats.seconds_elapsed = time.perf_counter() - t0
    return stats


def _import_root_for_repo(repo: Path) -> Path:
    """``sys.path`` entry that contains ``eqcct_tf_pt_transfer`` (bundle uses ``src/``)."""
    src = repo / "src"
    if (src / "eqcct_tf_pt_transfer").is_dir():
        return src.resolve()
    return repo.resolve()


def build_child_command(
    *,
    repo: Path,
    p_h5: Path,
    s_h5: Path,
    pt_p: Optional[Path],
    pt_s: Optional[Path],
    datasets: str,
    max_windows: int,
    stride: int,
    profiles_str: str,
    output_json: Optional[Path],
    no_tqdm: bool,
) -> list[str]:
    cmd: list[str] = [
        sys.executable,
        "-m",
        "eqcct_tf_pt_transfer.validation.tf_pt_seisbench_dataset_benchmark",
        "--repo",
        str(repo),
        "--p-h5",
        str(p_h5),
        "--s-h5",
        str(s_h5),
        "--datasets",
        datasets,
        "--max-windows",
        str(max_windows),
        "--stride",
        str(stride),
        "--profiles",
        profiles_str,
    ]
    if pt_p is not None:
        cmd.extend(["--pt-p", str(pt_p)])
    if pt_s is not None:
        cmd.extend(["--pt-s", str(pt_s)])
    if output_json is not None:
        cmd.extend(["--output-json", str(output_json)])
    if no_tqdm:
        cmd.append("--no-tqdm")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None, help="EQCCT_to_Seisbench root (default: cwd)")
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--pt-p", type=Path, default=None, help="Optional eqcct_model_p.pt")
    parser.add_argument("--pt-s", type=Path, default=None, help="Optional eqcct_model_s.pt")
    parser.add_argument(
        "--datasets",
        type=str,
        default="txed",
        help="Comma list: txed, stead, or both",
    )
    parser.add_argument("--max-windows", type=int, default=5000, help="Max qualifying windows per dataset (0 = no cap; can be huge)")
    parser.add_argument("--stride", type=int, default=1, help="Take every stride-th qualifying row from metadata")
    parser.add_argument(
        "--profiles",
        type=str,
        default="cpu,gpu0,gpu1",
        help="Comma-separated: cpu, gpu0, gpu1 (omit unavailable GPU profiles)",
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--no-tqdm", action="store_true")

    args = parser.parse_args()

    repo = args.repo or Path.cwd().resolve()
    if repo.name == "notebooks":
        repo = repo.parent
    ir = _import_root_for_repo(repo)
    if str(ir) not in sys.path:
        sys.path.insert(0, str(ir))

    p_h5 = args.p_h5 or repo / "ModelPS" / "test_trainer_024.h5"
    s_h5 = args.s_h5 or repo / "ModelPS" / "test_trainer_021.h5"
    pt_p = args.pt_p or (repo / "ModelPS" / "eqcct_model_p.pt")
    pt_s = args.pt_s or (repo / "ModelPS" / "eqcct_model_s.pt")
    if not pt_p.is_file():
        pt_p = None
    if not pt_s.is_file():
        pt_s = None

    ds_list: list[str]
    if args.datasets.lower() == "both":
        ds_list = ["txed", "stead"]
    else:
        ds_list = [x.strip().lower() for x in args.datasets.split(",") if x.strip()]

    import torch

    n_cuda = torch.cuda.device_count()
    profiles = parse_profiles(args.profiles, n_cuda)

    print("[info] repo:", repo)
    print("[info] datasets:", ds_list)
    print("[info] max_windows per dataset:", args.max_windows or "unlimited")
    print("[info] stride:", args.stride)
    print("[info] torch CUDA devices:", n_cuda)
    print("[info] profiles:", profiles)

    all_stats: list[dict] = []

    if len(profiles) > 1:
        child_env = os.environ.copy()
        child_env["EQCCT_TF_PT_QUIET_JSON"] = "1"
        repo_pp = str(_import_root_for_repo(repo))
        child_env["PYTHONPATH"] = (
            repo_pp + os.pathsep + child_env["PYTHONPATH"]
            if child_env.get("PYTHONPATH")
            else repo_pp
        )
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_path = Path(tmpd)
            for name, tf_dev, pt_dev in profiles:
                if args.output_json is not None:
                    child_out = args.output_json.with_name(
                        f"{args.output_json.stem}_{name}{args.output_json.suffix}"
                    )
                else:
                    child_out = tmp_path / f"tf_pt_benchmark_{name}.json"
                cmd = build_child_command(
                    repo=repo,
                    p_h5=p_h5,
                    s_h5=s_h5,
                    pt_p=pt_p,
                    pt_s=pt_s,
                    datasets=args.datasets,
                    max_windows=args.max_windows,
                    stride=args.stride,
                    profiles_str=name,
                    output_json=child_out,
                    no_tqdm=args.no_tqdm,
                )
                print(f"\n========== Profile {name} TF={tf_dev} PT={pt_dev} (subprocess) ==========")
                print("[info]", " ".join(cmd))
                subprocess.check_call(cmd, env=child_env, cwd=str(repo))
                payload = json.loads(child_out.read_text())
                all_stats.extend(payload.get("results", []))
                for d in payload.get("results", []):
                    print(json.dumps(d, indent=2))
    else:
        name, tf_dev, pt_dev = profiles[0]
        print(f"\n========== Profile {name} TF={tf_dev} PT={pt_dev} ==========")
        st = run_profile(
            name,
            tf_dev,
            pt_dev,
            datasets=ds_list,
            repo=repo,
            p_h5=p_h5,
            s_h5=s_h5,
            pt_p=pt_p,
            pt_s=pt_s,
            max_windows=args.max_windows,
            stride=args.stride,
            tqdm_disable=args.no_tqdm,
        )
        d = st.to_dict()
        all_stats.append(d)
        if not os.environ.get("EQCCT_TF_PT_QUIET_JSON"):
            print(json.dumps(d, indent=2))

    out = {
        "p_h5": str(p_h5),
        "s_h5": str(s_h5),
        "pt_p_ckpt": str(pt_p) if pt_p else None,
        "pt_s_ckpt": str(pt_s) if pt_s else None,
        "datasets": ds_list,
        "max_windows": args.max_windows,
        "stride": args.stride,
        "profiles": [p[0] for p in profiles],
        "results": all_stats,
    }
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(out, indent=2))
        print("\n[info] Wrote", args.output_json)


if __name__ == "__main__":
    main()
