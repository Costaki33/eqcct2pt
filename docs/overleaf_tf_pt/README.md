# Overleaf: TF–PyTorch weight transfer manuscript

This folder mirrors the structure used in `clean_eqcct/.../docs/overleaf/`: a small `main.tex` driver and `body.tex` content.

## Files to upload to Overleaf

| File | Purpose |
|------|---------|
| `main.tex` | Document class, packages, title, abstract, `\input{body}` |
| `body.tex` | Sections 1–7, acknowledgments, `\input{references}` |
| `references.tex` | Manual `\bibitem` bibliography (replace with `.bib` later if you prefer) |
| `figures/tf_pt_benchmark_tf_pt_discrepancy.png` | Figure 1 (benchmark bars); copy from repo `figures/` if you update the plot |
| `figures/tf_pt_layer_activations.png`           | Figure 2 (per-stage activation diff, P+S, CPU+GPU)                          |
| `figures/tf_pt_error_distributions.png`         | Figure 3 (per-window error CDFs / violins / boxplots)                       |
| `figures/tf_pt_waveform_overlays.png`           | Figure 4 (TXED + STEAD waveform overlays, TF vs PT)                         |
| `figures/tf_pt_perf_benchmark.png`              | Figure 5 (inference latency, throughput, RAM, VRAM)                         |

## Refresh the figures

From the repo root, run any of the data-collection commands you need (each one
caches its output under `results/`, then a small plot script renders the figure
into `figures/`). All collection commands isolate each device profile in its
own subprocess so TensorFlow's CPU mode does not hide GPUs for later profiles.

```bash
# Figure 1 — overall TF vs PT discrepancy bars (already cached)
python scripts/plot_tf_pt_benchmark_results.py

# Figure 2 — layer-by-layer activation discrepancy (P+S, CPU+GPU)
PYTHONPATH=. python -m eqcct_sb.validation.tf_pt_layer_activations \
  --profiles cpu,gpu0 --n-seeds 5 \
  --output-json results/layer_activations.json
python scripts/plot_layer_activations.py results/layer_activations.json

# Figure 3 — per-window error distributions (CDF / violin / box)
PYTHONPATH=. python -m eqcct_sb.validation.tf_pt_per_window_errors \
  --datasets txed,stead --max-windows 1000 --profiles cpu,gpu0 \
  --output-npz results/per_window_errors.npz
python scripts/plot_error_distributions.py results/per_window_errors.npz

# Figure 4 — TXED + STEAD waveform overlays (TF vs PT)
PYTHONPATH=. python -m eqcct_sb.validation.tf_pt_waveform_compare_figure \
  --datasets txed,stead --n-per-dataset 3 --seed 0 \
  --output figures/tf_pt_waveform_overlays.png

# Figure 5 — performance benchmark (latency, RAM, VRAM)
PYTHONPATH=. python -m eqcct_sb.validation.tf_pt_perf_benchmark \
  --profiles cpu,gpu0,gpu1 --n-windows 10 --warmup 3 \
  --output-json results/perf_benchmark.json
python scripts/plot_perf_benchmark.py results/perf_benchmark.json

# Then copy whatever you regenerated into the Overleaf folder
cp figures/tf_pt_benchmark_tf_pt_discrepancy.png \
   figures/tf_pt_layer_activations.png \
   figures/tf_pt_error_distributions.png \
   figures/tf_pt_waveform_overlays.png \
   figures/tf_pt_perf_benchmark.png \
   docs/overleaf_tf_pt/figures/
```

## Compile

Standard pdfLaTeX. Overleaf: set main document to `main.tex` and compile.

## Optional: match EQCCTPro exactly

If you want identical preamble to `clean_eqcct/eqcct/eqcctpro/docs/overleaf/main.tex`, copy `pdflscape` / `rotating` from that file; the current preamble is a minimal subset sufficient for this paper.
