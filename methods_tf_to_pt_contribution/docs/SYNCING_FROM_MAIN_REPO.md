# Keeping this bundle aligned with the main repository

The authoritative source lives in the **EQCCT / eqcct2pt** repo at the paths below (directories sit at the **repository root**, not nested under another package folder):

- `models/predictor_pt_p.py`
- `conversion/loader.py`
- `conversion/transfer_weights_legacy.py`
- `reference/predictor_tf.py`
- `validation/tf_pt_p_trace.py`
- `validation/tf_pt_s_trace.py`
- `validation/tf_pt_seisbench_dataset_benchmark.py`

## Refresh procedure

1. Copy each file into `methods_tf_to_pt_contribution/src/eqcct_tf_pt_transfer/` at the
   matching relative path (`models/`, `conversion/`, `reference/`, `validation/`).
2. Prefix imports with `eqcct_tf_pt_transfer` (see existing bundle files): e.g.
   `from eqcct_tf_pt_transfer.models...` rather than bare `models`.
3. Re-apply bundle-specific edits (do **not** overwrite blindly):
   - `paths.py`: resolves sibling `../ModelPS` Monorepo or `bundle/ModelPS` standalone.
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
