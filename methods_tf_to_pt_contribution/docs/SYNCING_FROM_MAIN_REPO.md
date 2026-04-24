# Keeping this bundle aligned with `eqcct_sb`

The authoritative source for these scripts is the parent project under:

- `eqcct_sb/models/predictor_pt_p.py`
- `eqcct_sb/conversion/loader.py`
- `eqcct_sb/conversion/transfer_weights_legacy.py`
- `eqcct_sb/reference/predictor_tf.py`
- `eqcct_sb/validation/tf_pt_p_trace.py`
- `eqcct_sb/validation/tf_pt_s_trace.py`
- `eqcct_sb/validation/tf_pt_seisbench_dataset_benchmark.py`

## Refresh procedure

1. Copy each file into `methods_tf_to_pt_contribution/src/eqcct_tf_pt_transfer/` at the
   matching relative path (`models/`, `conversion/`, `reference/`, `validation/`).
2. Replace the import prefix `eqcct_sb` → `eqcct_tf_pt_transfer` in every copied file.
3. Re-apply bundle-specific edits (do **not** overwrite blindly):
   - `validation/tf_pt_p_trace.py` and `tf_pt_s_trace.py`: `_repo_root()` must use
     `parents[3]` so default `ModelPS/` resolves to the bundle root.
   - `validation/tf_pt_seisbench_dataset_benchmark.py`: keep `_import_root_for_repo()`
     so `sys.path` uses `…/src` when the package lives under `src/`.
   - `reference/predictor_tf.py`: optional removal of unused imports (e.g. `ray`) if
     still unused in upstream.
4. Run a quick import check:

   ```bash
   cd methods_tf_to_pt_contribution
   PYTHONPATH=src python -c "import eqcct_tf_pt_transfer.conversion.loader as L; print('ok')"
   ```

5. Update `README.md` / `docs/` if workflows or flags changed.
