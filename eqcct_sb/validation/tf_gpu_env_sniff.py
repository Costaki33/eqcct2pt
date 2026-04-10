#!/usr/bin/env python3
"""
Quick check: does TensorFlow see GPUs in *this* Python interpreter?

Run from repo root:
  PYTHONPATH=/path/to/EQCCT_to_Seisbench python -m eqcct_sb.validation.tf_gpu_env_sniff

Use the same command in the terminal where `eqcctpro` works, and compare
`sys.executable` to the Jupyter kernel (printed at the start of the GPU
compare notebooks). If they differ, change the notebook kernel to that env.
"""
from __future__ import annotations

import os
import subprocess
import sys


def main() -> None:
    print("sys.executable:", sys.executable)
    print("sys.prefix:", sys.prefix)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))

    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        print("--- nvidia-smi -L ---")
        print(out.stdout.strip() or out.stderr.strip() or "(no output)")
    except FileNotFoundError:
        print("nvidia-smi: not found on PATH")
    except Exception as e:
        print("nvidia-smi error:", e)

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf

    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("tf.config.list_physical_devices('GPU'):", gpus)
    if not gpus:
        print(
            "\nNo GPUs — if nvidia-smi works elsewhere, this interpreter is likely "
            "not the conda env where TensorFlow was installed with GPU support. "
            "In Jupyter/Cursor: Kernel → pick the env whose path matches `sys.executable` "
            "from a working terminal.\n"
            "Register a kernel:  python -m ipykernel install --user --name eqcctpro "
            '--display-name "Python (eqcctpro)"'
        )
        sys.exit(1)
    print("OK:", len(gpus), "GPU(s) visible to TensorFlow.")
    sys.exit(0)


if __name__ == "__main__":
    main()
