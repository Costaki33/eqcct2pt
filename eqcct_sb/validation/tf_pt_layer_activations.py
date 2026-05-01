#!/usr/bin/env python3
"""
Layer-by-layer activation diff TF vs PT (P + S branches), for one or more device
profiles. For each named checkpoint along the architecture (front-end conv stack
-> patch encoder -> transformer blocks -> final LN -> picker reshape) we compute

  max |TF - PT|  and  mean |TF - PT|

averaged across N random ``(B, 6000, 3)`` inputs (default 5). Results for each
profile are dumped to JSON for plotting by ``scripts/plot_layer_activations.py``.

Each profile runs in its own subprocess (TF CPU mode hides GPUs globally and
would otherwise break later GPU profiles).

Example::

  PYTHONPATH=. python -m eqcct_sb.validation.tf_pt_layer_activations \\
    --profiles cpu,gpu0 --output-json results/layer_activations.json
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

from eqcct_sb.paths import MODELPS_DIR, REPO_ROOT


def _short_name(key: str) -> str:
    """Compact label for plotting (e.g. ``post_transformer_block2`` -> ``TB2``)."""
    if "post_front" in key:
        return "Conv front-end"
    if "post_patch_encoder" in key:
        return "Patch encoder"
    if "transformer_block" in key:
        return "TB" + key[-1]
    if "final_LN" in key:
        return "Final LN"
    if "after_reshape" in key:
        return "Pre-picker reshape"
    return key


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


def run_profile(
    profile: str,
    tf_device: str,
    torch_device: str,
    *,
    p_h5: Path,
    s_h5: Path,
    pt_p: Optional[Path],
    pt_s: Optional[Path],
    n_seeds: int,
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
    from eqcct_sb.validation.tf_pt_p_trace import _TF_ACTIVATION_STAGES as P_STAGES
    from eqcct_sb.validation.tf_pt_s_trace import _TF_ACTIVATION_STAGES as S_STAGES

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

    def _stage_diffs(model_tf, m_pt, stages, branch: str) -> list[dict]:
        out: list[dict] = []
        per_seed_max: dict[str, list[float]] = {st.key: [] for st in stages}
        per_seed_mean: dict[str, list[float]] = {st.key: [] for st in stages}
        for seed in range(n_seeds):
            rng = np.random.default_rng(seed)
            x = rng.standard_normal((1, 6000, 3)).astype(np.float32)
            x -= x.mean(axis=1, keepdims=True)
            x /= x.std(axis=1, keepdims=True) + 1e-8
            xt = torch.from_numpy(x).to(torch_device)
            for st in stages:
                lay = model_tf.layers[st.tf_layer_index]
                sub = tf.keras.Model(model_tf.input, lay.output)
                with tf.device(tf_device):
                    y_tf_t = sub(x, training=False)
                y_tf = y_tf_t.numpy() if hasattr(y_tf_t, "numpy") else np.asarray(y_tf_t)
                with torch.no_grad():
                    y_pt = st.pt_fn(m_pt, xt)
                a = st.tf_to_common(np.asarray(y_tf))
                b = st.pt_to_common(y_pt)
                if a.shape != b.shape:
                    per_seed_max[st.key].append(float("nan"))
                    per_seed_mean[st.key].append(float("nan"))
                    continue
                d = np.abs(a - b)
                per_seed_max[st.key].append(float(d.max()))
                per_seed_mean[st.key].append(float(d.mean()))
        for st in stages:
            mx = np.asarray(per_seed_max[st.key])
            me = np.asarray(per_seed_mean[st.key])
            out.append({
                "key": st.key,
                "short": _short_name(st.key),
                "branch": branch,
                "tf_layer_index": st.tf_layer_index,
                "max_abs_diff_mean": float(np.nanmean(mx)),
                "max_abs_diff_std": float(np.nanstd(mx, ddof=1)) if mx.size > 1 else 0.0,
                "mean_abs_diff_mean": float(np.nanmean(me)),
                "mean_abs_diff_std": float(np.nanstd(me, ddof=1)) if me.size > 1 else 0.0,
                "n_seeds": int(mx.size),
            })
        return out

    p_records = _stage_diffs(model_p_tf, m_p, P_STAGES, branch="P")
    s_records = _stage_diffs(model_s_tf, m_s, S_STAGES, branch="S")

    return {
        "profile": profile,
        "tf_device": tf_device,
        "torch_device": torch_device,
        "n_seeds": n_seeds,
        "stages_p": p_records,
        "stages_s": s_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=None)
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--s-h5", type=Path, default=None)
    parser.add_argument("--pt-p", type=Path, default=None)
    parser.add_argument("--pt-s", type=Path, default=None)
    parser.add_argument("--profiles", type=str, default="cpu,gpu0")
    parser.add_argument("--n-seeds", type=int, default=5)
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
                child_out = tmp_path / f"act_{name}.json"
                cmd = [
                    sys.executable,
                    "-m",
                    "eqcct_sb.validation.tf_pt_layer_activations",
                    "--repo", str(repo),
                    "--p-h5", str(p_h5),
                    "--s-h5", str(s_h5),
                    "--profiles", name,
                    "--n-seeds", str(args.n_seeds),
                    "--output-json", str(child_out),
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
            n_seeds=args.n_seeds,
        )
        results.append(d)
        if not os.environ.get("EQCCT_TF_PT_QUIET_JSON"):
            print(json.dumps(d, indent=2))

    out = {
        "p_h5": str(p_h5),
        "s_h5": str(s_h5),
        "pt_p_ckpt": str(pt_p) if pt_p else None,
        "pt_s_ckpt": str(pt_s) if pt_s else None,
        "n_seeds": args.n_seeds,
        "profiles": [p[0] for p in profiles],
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2))
    print("\n[info] Wrote", args.output_json)


if __name__ == "__main__":
    main()
