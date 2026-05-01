#!/usr/bin/env python3
"""
TF vs PT performance benchmark: inference time, host RAM and GPU VRAM, throughput.

For each device profile (``cpu``, ``gpu0``, ``gpu1``) the script

* loads ``modelP`` / ``modelS`` (TensorFlow Keras), measures host RAM delta and TF
  peak GPU memory (if applicable);
* loads ``EQCCTModelP`` / ``EQCCTModelS`` (PyTorch), measures the additional host
  RAM delta and PyTorch peak VRAM (if applicable);
* runs 10 different 60-second (6000 sample @ 100 Hz) ``(1, 6000, 3)`` windows
  through P then S for both backends after a small warmup, recording per-call
  inference time and computing throughput as windows / sec.

If multiple profiles are requested, each profile runs in a separate Python
subprocess so that TF CPU-only mode (which hides GPUs globally) does not break
subsequent GPU profiles. JSON output schema is documented at the bottom.

Example::

  PYTHONPATH=. python -m validation.tf_pt_perf_benchmark \\
    --profiles cpu,gpu0,gpu1 --output-json results/perf_benchmark.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import numpy as np

from paths import MODELPS_DIR, REPO_ROOT


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024**2)
    except Exception:
        return float("nan")


def _make_windows(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 6000, 3)).astype(np.float32)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True) + 1e-8
    return x


def _sync_gpu(torch_device: str) -> None:
    if torch_device.startswith("cuda"):
        try:
            import torch

            torch.cuda.synchronize(torch_device)
        except Exception:
            pass


def _tf_peak_gpu_mb(tf_device: str) -> Optional[float]:
    if not tf_device.startswith("/GPU"):
        return None
    try:
        import tensorflow as tf

        info = tf.config.experimental.get_memory_info(tf_device.replace("/", ""))
        return float(info.get("peak", 0)) / (1024**2)
    except Exception:
        return None


def _torch_peak_vram_mb(torch_device: str) -> Optional[float]:
    if not torch_device.startswith("cuda"):
        return None
    try:
        import torch

        idx = torch.device(torch_device).index
        if idx is None:
            idx = torch.cuda.current_device()
        return torch.cuda.max_memory_allocated(idx) / (1024**2)
    except Exception:
        return None


def run_profile(
    profile: str,
    tf_device: str,
    torch_device: str,
    *,
    p_h5: Path,
    s_h5: Path,
    pt_p: Optional[Path],
    pt_s: Optional[Path],
    n_windows: int,
    warmup: int,
) -> dict:
    """Single-profile measurement (must run in a fresh process)."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    import tensorflow as tf
    import torch

    if tf_device.startswith("/CPU"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    rss_baseline = _rss_mb()

    from reference.predictor_tf import load_eqcct_model
    from models.predictor_pt_p import EQCCTModelP, EQCCTModelS
    from conversion.loader import load_eqcct_model_p_weights, load_eqcct_model_s_weights

    t_load_tf0 = time.perf_counter()
    model_p_tf, model_s_tf = load_eqcct_model(str(p_h5), str(s_h5))
    t_load_tf = time.perf_counter() - t_load_tf0
    rss_after_tf = _rss_mb()

    t_load_pt0 = time.perf_counter()
    model_p_pt = EQCCTModelP().eval()
    model_s_pt = EQCCTModelS().eval()
    if pt_p and pt_p.is_file():
        ck = torch.load(pt_p, map_location="cpu", weights_only=False)
        model_p_pt.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_p_weights(model_p_pt, h5_path=str(p_h5))
    if pt_s and pt_s.is_file():
        ck = torch.load(pt_s, map_location="cpu", weights_only=False)
        model_s_pt.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_s_weights(model_s_pt, h5_path=str(s_h5))
    model_p_pt = model_p_pt.to(torch_device)
    model_s_pt = model_s_pt.to(torch_device)
    t_load_pt = time.perf_counter() - t_load_pt0
    rss_after_pt = _rss_mb()

    windows = _make_windows(n_windows + warmup, seed=42)

    # Warmup (excluded from timing)
    for w in range(warmup):
        x_tf = tf.constant(windows[w : w + 1], dtype=tf.float32)
        with tf.device(tf_device):
            _ = model_p_tf(x_tf, training=False)
            _ = model_s_tf(x_tf, training=False)
        x_pt = torch.from_numpy(windows[w : w + 1]).to(torch_device)
        with torch.no_grad():
            _ = model_p_pt(x_pt)
            _ = model_s_pt(x_pt)
    _sync_gpu(torch_device)

    # Reset TF/PT peak memory counters after warmup so we measure inference VRAM,
    # not initial alloc / cuDNN workspace probing.
    if tf_device.startswith("/GPU"):
        try:
            tf.config.experimental.reset_memory_stats(tf_device.replace("/", ""))
        except Exception:
            pass
    if torch_device.startswith("cuda"):
        try:
            idx = torch.device(torch_device).index
            if idx is None:
                idx = torch.cuda.current_device()
            torch.cuda.reset_peak_memory_stats(idx)
        except Exception:
            pass

    # Inference timings (per window)
    tf_p_times: list[float] = []
    tf_s_times: list[float] = []
    pt_p_times: list[float] = []
    pt_s_times: list[float] = []

    for i in range(warmup, warmup + n_windows):
        wf = windows[i : i + 1]

        x_tf = tf.constant(wf, dtype=tf.float32)
        with tf.device(tf_device):
            t0 = time.perf_counter()
            out = model_p_tf(x_tf, training=False)
            if hasattr(out, "numpy"):
                out.numpy()
            tf_p_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            out = model_s_tf(x_tf, training=False)
            if hasattr(out, "numpy"):
                out.numpy()
            tf_s_times.append(time.perf_counter() - t0)

        x_pt = torch.from_numpy(wf).to(torch_device)
        with torch.no_grad():
            _sync_gpu(torch_device)
            t0 = time.perf_counter()
            _ = model_p_pt(x_pt)
            _sync_gpu(torch_device)
            pt_p_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _ = model_s_pt(x_pt)
            _sync_gpu(torch_device)
            pt_s_times.append(time.perf_counter() - t0)

    tf_peak_vram = _tf_peak_gpu_mb(tf_device)
    pt_peak_vram = _torch_peak_vram_mb(torch_device)

    def _stats(xs: list[float]) -> dict:
        a = np.asarray(xs, dtype=np.float64)
        return {
            "mean_s": float(a.mean()),
            "median_s": float(np.median(a)),
            "std_s": float(a.std(ddof=1)) if a.size > 1 else 0.0,
            "min_s": float(a.min()),
            "max_s": float(a.max()),
            "throughput_win_per_s": float(1.0 / a.mean()),
            "all_s": [float(x) for x in a],
        }

    return {
        "profile": profile,
        "tf_device": tf_device,
        "torch_device": torch_device,
        "n_windows": n_windows,
        "warmup": warmup,
        "rss_baseline_mb": rss_baseline,
        "rss_after_tf_load_mb": rss_after_tf,
        "rss_after_pt_load_mb": rss_after_pt,
        "rss_tf_load_delta_mb": rss_after_tf - rss_baseline,
        "rss_pt_load_delta_mb": rss_after_pt - rss_after_tf,
        "rss_total_loaded_delta_mb": rss_after_pt - rss_baseline,
        "tf_load_seconds": t_load_tf,
        "pt_load_seconds": t_load_pt,
        "tf_peak_vram_mb": tf_peak_vram,
        "pt_peak_vram_mb": pt_peak_vram,
        "tf_p": _stats(tf_p_times),
        "tf_s": _stats(tf_s_times),
        "pt_p": _stats(pt_p_times),
        "pt_s": _stats(pt_s_times),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--pt-p", type=Path, default=None)
    parser.add_argument("--pt-s", type=Path, default=None)
    parser.add_argument("--profiles", type=str, default="cpu,gpu0")
    parser.add_argument("--n-windows", type=int, default=10, help="Number of timed windows per backend")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output-json", type=Path, required=True)
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

    import torch

    n_cuda = torch.cuda.device_count()
    profiles = parse_profiles(args.profiles, n_cuda)
    if not profiles:
        raise SystemExit("No valid profiles requested.")

    results: list[dict] = []
    if len(profiles) > 1:
        child_env = os.environ.copy()
        child_env["EQCCT_TF_PT_QUIET_JSON"] = "1"
        repo_pp = str(repo)
        child_env["PYTHONPATH"] = (
            repo_pp + os.pathsep + child_env["PYTHONPATH"]
            if child_env.get("PYTHONPATH")
            else repo_pp
        )
        with tempfile.TemporaryDirectory() as tmpd:
            tmp_path = Path(tmpd)
            for name, tf_dev, pt_dev in profiles:
                child_out = tmp_path / f"perf_{name}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "validation.tf_pt_perf_benchmark",
                    "--repo",
                    str(repo),
                    "--p-h5",
                    str(p_h5),
                    "--s-h5",
                    str(s_h5),
                    "--profiles",
                    name,
                    "--n-windows",
                    str(args.n_windows),
                    "--warmup",
                    str(args.warmup),
                    "--output-json",
                    str(child_out),
                ]
                if pt_p is not None:
                    cmd.extend(["--pt-p", str(pt_p)])
                if pt_s is not None:
                    cmd.extend(["--pt-s", str(pt_s)])
                print(f"\n========== Profile {name} (subprocess) ==========")
                print("[info]", " ".join(cmd))
                subprocess.check_call(cmd, env=child_env, cwd=str(repo))
                results.extend(json.loads(child_out.read_text())["results"])
    else:
        name, tf_dev, pt_dev = profiles[0]
        d = run_profile(
            name, tf_dev, pt_dev,
            p_h5=p_h5, s_h5=s_h5, pt_p=pt_p, pt_s=pt_s,
            n_windows=args.n_windows, warmup=args.warmup,
        )
        results.append(d)
        if not os.environ.get("EQCCT_TF_PT_QUIET_JSON"):
            print(json.dumps(d, indent=2))

    out = {
        "p_h5": str(p_h5),
        "s_h5": str(s_h5),
        "pt_p_ckpt": str(pt_p) if pt_p else None,
        "pt_s_ckpt": str(pt_s) if pt_s else None,
        "n_windows": args.n_windows,
        "warmup": args.warmup,
        "profiles": [p[0] for p in profiles],
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print("\n[info] Wrote", args.output_json)


if __name__ == "__main__":
    main()
