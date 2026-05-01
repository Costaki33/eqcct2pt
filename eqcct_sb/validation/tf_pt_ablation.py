#!/usr/bin/env python3
"""
Ablation study: TF vs three PT variants on real SeisBench windows.

The goal is to show that the common failure modes during TF -> PT weight
transfer are **catastrophic**, not subtle -- so a careful validation is
necessary.  Three PT variants are compared to TensorFlow:

  A) ``correct``        -- fully validated PT model (baseline for parity)
  B) ``no_transpose``   -- the Patch Encoder Dense layer is loaded *without*
                           the Keras->PyTorch (in, out) -> (out, in) transpose.
                           Because Keras Dense kernels and PyTorch Linear weights
                           have different conventions, and the Patch Encoder is
                           square, this silently misorders elements without
                           raising a shape error.
  C) ``missing_picker`` -- the ``picker_P`` / ``picker_S`` Conv1D heads are
                           left at PyTorch's default random initialisation
                           (simulating a weight-loader that skips the picker,
                           e.g. ``load_state_dict(strict=False)`` after a
                           partial checkpoint, or Keras ``by_name=True`` losing
                           the picker layer).

For each selected window we record:

  * MAE per window (P and S) for each variant
  * Pick-time error in samples (|argmax(TF) - argmax(PT)|) for each variant

We also save a handful of example waveforms + TF / PT P probability traces
so the plotter can render the qualitative top row of Figure 6.

Each device profile (``cpu``, ``gpu0``, ``gpu1``) runs in its own subprocess
(TF CPU mode hides GPUs globally and would otherwise break later GPU profiles).

Output: a single NPZ.  Keys:

  * ``<profile>_<variant>_<metric>`` for ``metric in {mae_p, mae_s, pterr, sterr}``
    and ``variant in {correct, no_transpose, missing_picker}``
  * ``<profile>_example_<i>_wf``, ``<profile>_example_<i>_p_tf``,
    ``<profile>_example_<i>_p_<variant>`` for ``i in 0..n_examples-1``
  * ``<profile>_example_<i>_dataset``, ``<profile>_example_<i>_trace``

Example::

  PYTHONPATH=. python -m eqcct_sb.validation.tf_pt_ablation \\
    --profiles cpu --max-windows 1000 --n-examples 3 \\
    --output-npz results/ablation.npz
"""
from __future__ import annotations

import argparse
import copy
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from eqcct_sb.paths import MODELPS_DIR, REPO_ROOT

VARIANTS = ("correct", "no_transpose", "missing_picker")


def parse_profiles(s: str, n_cuda: int):
    out = []
    for part in s.split(","):
        part = part.strip().lower()
        if part == "cpu":
            out.append(("cpu", "/CPU:0", "cpu"))
        elif part in ("gpu0", "cuda0", "0"):
            if n_cuda >= 1:
                out.append(("gpu0", "/GPU:0", "cuda:0"))
        elif part in ("gpu1", "cuda1", "1"):
            if n_cuda >= 2:
                out.append(("gpu1", "/GPU:1", "cuda:1"))
        else:
            raise ValueError(part)
    return out


def _norm_std_time(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True) + 1e-8
    return ((x - m) / s).astype(np.float32)


def _build_window(dataset, idx: int, p_sample: int, s_sample: int) -> Optional[np.ndarray]:
    raw = np.asarray(dataset.get_waveforms([idx])[0], dtype=np.float32)
    if raw.shape[1] < 6000:
        return None
    start = max(0, min(p_sample - 3000, raw.shape[1] - 6000))
    s_in = s_sample - start
    if not (0 <= s_in < 6000):
        return None
    win = raw[:, start : start + 6000]
    return _norm_std_time(win.transpose(1, 0)[np.newaxis, ...])


def _make_variant_no_transpose(correct_model):
    """Clone a correct PT model and undo the Patch Encoder transpose (silently broken)."""
    import torch
    bad = copy.deepcopy(correct_model)
    W = bad.encoder.projection.weight.data
    if W.shape[0] != W.shape[1]:
        print(f"[warn] Patch Encoder is not square ({tuple(W.shape)}); "
              f"no-transpose variant would crash, skipping the transpose flip.")
    else:
        bad.encoder.projection.weight.data = W.t().contiguous().clone()
    return bad


def _make_variant_missing_picker(correct_model, seed: int = 0):
    """Clone a correct PT model and re-initialise the picker head randomly."""
    import torch
    bad = copy.deepcopy(correct_model)
    for name in ("picker_P", "picker_S"):
        head = getattr(bad, name, None)
        if head is None:
            continue
        g = torch.Generator().manual_seed(seed + (0 if name == "picker_P" else 1))
        if hasattr(head, "weight") and head.weight is not None:
            fan_in = head.weight.numel() // head.weight.shape[0]
            std = (2.0 / fan_in) ** 0.5
            head.weight.data = torch.empty_like(head.weight).normal_(0.0, std, generator=g)
        if hasattr(head, "bias") and head.bias is not None:
            head.bias.data.zero_()
    return bad


def _argmax_sample(curve: np.ndarray) -> int:
    return int(np.argmax(curve.ravel()))


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
    n_examples: int,
    seed: int,
) -> dict:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    import torch

    if tf_device.startswith("/CPU"):
        try:
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass

    from eqcct_sb.reference.predictor_tf import load_eqcct_model
    from eqcct_sb.models.predictor_pt_p import EQCCTModelP, EQCCTModelS
    from eqcct_sb.conversion.loader import load_eqcct_model_p_weights, load_eqcct_model_s_weights

    model_p_tf, model_s_tf = load_eqcct_model(str(p_h5), str(s_h5))

    m_p_ok = EQCCTModelP().eval()
    m_s_ok = EQCCTModelS().eval()
    if pt_p and pt_p.is_file():
        ck = torch.load(pt_p, map_location="cpu", weights_only=False)
        m_p_ok.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_p_weights(m_p_ok, h5_path=str(p_h5))
    if pt_s and pt_s.is_file():
        ck = torch.load(pt_s, map_location="cpu", weights_only=False)
        m_s_ok.load_state_dict(ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck)
    else:
        load_eqcct_model_s_weights(m_s_ok, h5_path=str(s_h5))

    m_p_nt = _make_variant_no_transpose(m_p_ok)
    m_s_nt = _make_variant_no_transpose(m_s_ok)
    m_p_mp = _make_variant_missing_picker(m_p_ok, seed=seed)
    m_s_mp = _make_variant_missing_picker(m_s_ok, seed=seed + 2)

    pt_p_models = {"correct": m_p_ok, "no_transpose": m_p_nt, "missing_picker": m_p_mp}
    pt_s_models = {"correct": m_s_ok, "no_transpose": m_s_nt, "missing_picker": m_s_mp}
    for mdl in (*pt_p_models.values(), *pt_s_models.values()):
        mdl.to(torch_device).eval()

    try:
        import seisbench.data as sbd
    except ImportError as e:
        raise SystemExit("seisbench is required: pip install seisbench") from e
    try:
        from tqdm import tqdm
    except ImportError:
        def tqdm(x, **kwargs):
            return x

    metrics = {v: {"mae_p": [], "mae_s": [], "pterr": [], "sterr": []} for v in VARIANTS}
    examples: list[dict] = []
    rng = np.random.default_rng(seed)

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
        ex_budget = max(0, n_examples - len(examples))
        ex_positions = set()
        if ex_budget > 0 and len(indices) > 0:
            ex_positions = set(rng.choice(len(indices), size=min(ex_budget, len(indices)), replace=False).tolist())

        for pos, j in enumerate(tqdm(indices, desc=f"{profile}/{ds_name}", file=sys.stdout)):
            row = sub.iloc[j]
            try:
                idx = ds.get_idx_from_trace_name(row["trace_name"])
                wf = _build_window(ds, idx, int(row["trace_p_arrival_sample"]), int(row["trace_s_arrival_sample"]))
            except Exception:
                continue
            if wf is None:
                continue
            try:
                p_tf_t = model_p_tf(tf.constant(wf, dtype=tf.float32), training=False)
                s_tf_t = model_s_tf(tf.constant(wf, dtype=tf.float32), training=False)
                p_tf = p_tf_t.numpy() if hasattr(p_tf_t, "numpy") else np.asarray(p_tf_t)
                s_tf = s_tf_t.numpy() if hasattr(s_tf_t, "numpy") else np.asarray(s_tf_t)
                if p_tf.ndim == 3:
                    p_tf = p_tf[..., 0]
                if s_tf.ndim == 3:
                    s_tf = s_tf[..., 0]
                p_tf = p_tf[0]
                s_tf = s_tf[0]
                tf_pmax = _argmax_sample(p_tf)
                tf_smax = _argmax_sample(s_tf)

                per_variant_curves = {}
                with torch.no_grad():
                    x_pt = torch.from_numpy(wf).to(torch_device)
                    for v in VARIANTS:
                        p_v = pt_p_models[v](x_pt).detach().cpu().numpy()[..., 0][0]
                        s_v = pt_s_models[v](x_pt).detach().cpu().numpy()[..., 0][0]
                        metrics[v]["mae_p"].append(float(np.abs(p_tf - p_v).mean()))
                        metrics[v]["mae_s"].append(float(np.abs(s_tf - s_v).mean()))
                        metrics[v]["pterr"].append(abs(tf_pmax - _argmax_sample(p_v)))
                        metrics[v]["sterr"].append(abs(tf_smax - _argmax_sample(s_v)))
                        per_variant_curves[v] = (p_v, s_v)
            except Exception:
                continue

            if pos in ex_positions and len(examples) < n_examples:
                examples.append({
                    "dataset": ds_name,
                    "trace": row["trace_name"],
                    "wf": wf[0].copy(),
                    "p_tf": p_tf.copy(),
                    "s_tf": s_tf.copy(),
                    "p_curves": {v: c[0].copy() for v, c in per_variant_curves.items()},
                    "s_curves": {v: c[1].copy() for v, c in per_variant_curves.items()},
                })

    out = {"profile": profile}
    for v in VARIANTS:
        for k, arr in metrics[v].items():
            out[f"{v}_{k}"] = np.asarray(arr, dtype=np.float64)
    out["examples"] = examples
    return out


def _flatten_profile_arrays(profile: str, d: dict) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for v in VARIANTS:
        for metric in ("mae_p", "mae_s", "pterr", "sterr"):
            arrays[f"{profile}_{v}_{metric}"] = d[f"{v}_{metric}"]
    for i, ex in enumerate(d["examples"]):
        arrays[f"{profile}_example_{i}_wf"] = ex["wf"]
        arrays[f"{profile}_example_{i}_p_tf"] = ex["p_tf"]
        arrays[f"{profile}_example_{i}_s_tf"] = ex["s_tf"]
        for v in VARIANTS:
            arrays[f"{profile}_example_{i}_p_{v}"] = ex["p_curves"][v]
            arrays[f"{profile}_example_{i}_s_{v}"] = ex["s_curves"][v]
        arrays[f"{profile}_example_{i}_dataset"] = np.asarray(ex["dataset"])
        arrays[f"{profile}_example_{i}_trace"] = np.asarray(ex["trace"])
    return arrays


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--pt-p", type=Path, default=None)
    parser.add_argument("--pt-s", type=Path, default=None)
    parser.add_argument("--datasets", type=str, default="txed,stead")
    parser.add_argument("--max-windows", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profiles", type=str, default="cpu")
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--_child_profile", type=str, default=None,
                        help="internal: single-profile child run")
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

    if args.__dict__.get("_child_profile"):
        profs = parse_profiles(args.__dict__["_child_profile"], n_cuda)
        if not profs:
            raise SystemExit("No valid profiles in child call.")
        name, tf_dev, pt_dev = profs[0]
        d = run_profile(
            name, tf_dev, pt_dev,
            p_h5=p_h5, s_h5=s_h5, pt_p=pt_p, pt_s=pt_s,
            datasets=ds_list, max_windows=args.max_windows, stride=args.stride,
            n_examples=args.n_examples, seed=args.seed,
        )
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.output_npz, **_flatten_profile_arrays(name, d))
        print("[info] Wrote child npz", args.output_npz)
        return

    profiles = parse_profiles(args.profiles, n_cuda)
    if not profiles:
        raise SystemExit("No valid profiles.")

    merged: dict[str, np.ndarray] = {}
    with tempfile.TemporaryDirectory() as tmpd:
        tmp = Path(tmpd)
        child_env = os.environ.copy()
        repo_pp = str(repo)
        child_env["PYTHONPATH"] = (
            repo_pp + os.pathsep + child_env["PYTHONPATH"]
            if child_env.get("PYTHONPATH") else repo_pp
        )
        for name, _tf_dev, _pt_dev in profiles:
            child_npz = tmp / f"ablation_{name}.npz"
            cmd = [
                sys.executable, "-m", "eqcct_sb.validation.tf_pt_ablation",
                "--repo", str(repo),
                "--p-h5", str(p_h5), "--s-h5", str(s_h5),
                "--datasets", args.datasets,
                "--max-windows", str(args.max_windows),
                "--stride", str(args.stride),
                "--n-examples", str(args.n_examples),
                "--seed", str(args.seed),
                "--profiles", name,
                "--_child_profile", name,
                "--output-npz", str(child_npz),
            ]
            if pt_p is not None:
                cmd.extend(["--pt-p", str(pt_p)])
            if pt_s is not None:
                cmd.extend(["--pt-s", str(pt_s)])
            print(f"\n========== Ablation profile {name} (subprocess) ==========")
            print("[info]", " ".join(cmd))
            subprocess.check_call(cmd, env=child_env, cwd=str(repo))
            z = np.load(child_npz, allow_pickle=True)
            for k in z.files:
                merged[k] = z[k]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output_npz, **merged)
    print("[info] Wrote", args.output_npz)


if __name__ == "__main__":
    main()
