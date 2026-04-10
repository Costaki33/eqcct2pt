"""
EQCCT ↔ SeisBench port: TensorFlow reference, PyTorch models, and weight conversion.

Run scripts with repo root on PYTHONPATH, e.g.:
  PYTHONPATH=/path/to/EQCCT_to_Seisbench python -m eqcct_sb.training.finetune_s_model
"""

from eqcct_sb.models.predictor_pt import EQCCTModelP, EQCCTModelS

__all__ = ["EQCCTModelP", "EQCCTModelS"]
