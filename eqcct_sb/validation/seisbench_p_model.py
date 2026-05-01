#!/usr/bin/env python3
"""
Smoke test: load SeisBench data (STEAD or TXED), run EQCCTModelP with P-phase labels.

Run from repo root (downloads/cache handled by SeisBench):
  PYTHONPATH=/path/to/EQCCT_to_Seisbench python -m eqcct_sb.validation.seisbench_p_model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eqcct_sb.conversion.loader import load_eqcct_model_p_weights
from eqcct_sb.models.predictor_pt_p import EQCCTModelP
from eqcct_sb.paths import MODELPS_DIR


def _build_dataloader(dataset: str, sampling_rate: int, batch_size: int, num_workers: int):
    import seisbench.data as sbd
    import seisbench.generate as sbg

    if dataset.lower() == "stead":
        data = sbd.STEAD(sampling_rate=sampling_rate, component_order="ZNE")
    elif dataset.lower() == "txed":
        data = sbd.TXED(sampling_rate=sampling_rate, component_order="ZNE")
    else:
        raise ValueError("dataset must be stead or txed")

    # STEAD & TXED expose this column in metadata (lowercase p).
    phase_dict = {"trace_p_arrival_sample": "P"}
    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=2000,
            windowlen=6000,
            selection="first",
            strategy="pad",
        ),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict,
            model_labels="P",
            dim=0,
            sigma=10,
            shape="gaussian",
        ),
        sbg.Normalize(
            demean_axis=-1,
            amp_norm_axis=-1,
            amp_norm_type="peak",
            key=("X", "X"),
        ),
        sbg.ChangeDtype(np.float32),
    ]

    try:
        data.preload_waveforms(p_snr=20.0)
    except TypeError:
        try:
            data.preload_waveforms()
        except Exception:
            pass

    train_split, _, _ = data.train_dev_test()
    generator = sbg.GenericGenerator(train_split)
    generator.add_augmentations(augmentations)

    loader_kw: dict = {"shuffle": False}
    if num_workers and num_workers > 0:
        loader_kw["num_workers"] = num_workers
        loader_kw["persistent_workers"] = True

    return DataLoader(generator, batch_size=batch_size, **loader_kw)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="SeisBench batch smoke test for EQCCTModelP")
    parser.add_argument("--dataset", choices=("stead", "txed"), default="txed")
    parser.add_argument("--p-h5", type=Path, default=None)
    parser.add_argument("--sampling-rate", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    p_h5 = args.p_h5 or MODELPS_DIR / "test_trainer_024.h5"
    if not p_h5.is_file():
        print(f"Missing P weights: {p_h5}", file=sys.stderr)
        return 2

    try:
        loader = _build_dataloader(
            args.dataset,
            args.sampling_rate,
            args.batch_size,
            args.num_workers,
        )
    except ImportError as e:
        print(f"SeisBench not installed: {e}", file=sys.stderr)
        return 3
    except Exception as e:
        print(
            "Failed to build SeisBench dataloader (dataset cached / path / columns). "
            f"Details: {e}",
            file=sys.stderr,
        )
        return 3

    model = EQCCTModelP().to(args.device)
    load_eqcct_model_p_weights(model, h5_path=str(p_h5))
    model.eval()

    print(f"Running {args.num_batches} batch(es) from {args.dataset.upper()}…")
    it = iter(loader)
    for b in range(args.num_batches):
        try:
            batch = next(it)
        except StopIteration:
            print("Ran out of batches.", file=sys.stderr)
            break
        waveforms = batch["X"].to(args.device)
        if waveforms.ndim != 3 or waveforms.shape[1] != 3:
            print(f"Unexpected X shape {waveforms.shape}", file=sys.stderr)
            return 4
        waveforms = waveforms.permute(0, 2, 1)
        y = batch.get("y")
        with torch.no_grad():
            out = model(waveforms)
        print(
            f"  batch {b}: X {tuple(waveforms.shape)} -> out {tuple(out.shape)} "
            f"mean={out.mean().item():.5f} std={out.std().item():.5f}"
        )
        if y is not None:
            y = y.to(args.device)
            if y.ndim == 3:
                print(f"           y {tuple(y.shape)}")
    print("[ok] SeisBench forward smoke test finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
