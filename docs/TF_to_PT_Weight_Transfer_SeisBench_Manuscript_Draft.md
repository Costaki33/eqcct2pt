# Porting institutional deep-learning pickers to PyTorch and SeisBench: a practical guide to TensorFlow–PyTorch weight transfer (EQCCT case study)

**Authors:** C. Skevofilax, A. Savvaidis

**Affiliations:** The University of Texas at Austin

**Keywords:** seismic phase picking; deep learning; model portability; TensorFlow; PyTorch; SeisBench; EQCCT; weight transfer; reproducibility

---

## Abstract

Seismic networks and research groups increasingly train in-house phase-picking models with the machine-learning framework that best fits their workflow—often **TensorFlow / Keras**. At the same time, much of the modern open-source ecosystem, including the widely used toolbox **SeisBench**, has coalesced around **PyTorch**: recent surveys of the deep-learning literature report PyTorch in the large majority of new studies (on the order of **~85%** of papers in some yearly samples), and SeisBench itself centers on **PyTorch Lightning** models, shared `from_pretrained` checkpoints, and a uniform interface for benchmarking and reuse. That mismatch leaves a practical gap: weights trained in TensorFlow are not automatically interchangeable with the PyTorch-first tooling that many networks now want for **benchmarking, retraining, community sharing, and integration** with orchestration stacks. Without a **reproducible** path from the original training framework to PyTorch, institutional models cannot be distributed as first-class citizens alongside SeisBench-native pickers, and downstream workflows remain fragile.

This manuscript presents a **manual-style workflow** for transferring **EQCCT**-style pickers from **TensorFlow** to **PyTorch**, using our open implementation as a case study. We document recurring pitfalls—**Keras** `load_weights` ordering, **dense-layer** layout when matrix dimensions are square, a **missing S-branch picker head** in early loaders, and **TensorFlow GPU / XLA** behavior when judging numerical agreement—and we show how we **validated** parity using layer-wise weight and activation traces plus end-to-end checks on real waveforms. We briefly connect these results to **network-scale deployment**: orchestration frameworks such as **EQCCTPro / RAPID** gain the most from **persistent** model workers when the loaded weights provably match the original trainer. The intended audience is developers who maintain non-PyTorch pickers and wish to publish **compatible checkpoints** and SeisBench-ready implementations **without retraining from scratch**.

---

## 1. Introduction

### 1.1 Motivation

Deep-learning phase pickers (e.g., PhaseNet, EQTransformer, EQCCT) have improved automated catalogs and real-time products. **SeisBench** (Woollam et al., 2022) standardizes data access, model interfaces, and pretrained weights for many architectures, lowering the barrier to comparative studies. In parallel, operational and experimental systems—**SeisComP**, custom ingest, and high-throughput orchestration frameworks—need models that (i) match the **original** training-time behavior, (ii) run efficiently under **memory and latency** constraints, and (iii) can be **shared** with collaborators in a single, well-documented format.

Institutional groups often train models in **TensorFlow** (as with published EQCCT training pipelines). SeisBench’s native model zoo is **PyTorch-centric**. A gap therefore appears: *How does a network deliver an EQCCT-class model to the community in a PyTorch-compatible form without discarding years of tuned weights?*

### 1.2 Scope of this paper

This is **not** a new picking architecture. It is a **methods** contribution: a reproducible recipe to:

1. Map **Keras HDF5** checkpoints to **PyTorch** `state_dict` entries.
2. Prove **numerical parity** between TensorFlow and PyTorch forward passes.
3. Run **SeisBench-style** waveform windows for qualitative validation.
4. Align with **deployment** contexts where TensorFlow re-initialization per task is costly (e.g., ephemeral Ray tasks vs. persistent actors; see EQCCTPro / RAPID discussion in Saad et al., 2023; Chen et al., 2024; and parallel orchestration work in the EQCCTPro draft).

We use **EQCCT** (separate **P** and **S** branches, Keras `modelP` / `modelS`) as a **case study** because it exemplifies a TensorFlow-only path in operational writing and because its **S** branch adds extra convolutional blocks relative to **P**, stressing correct **layer ordering** in the loader.

### 1.3 Reader takeaway

After reading this draft, a practitioner should be able to:

- Structure a **flat weight dictionary** from Keras `.h5` files.
- Implement or audit `load_*_weights` functions that pair **sorted** layer names with **fixed** module lists.
- Run **automated** TF vs. PT comparisons (`parity_*`, `tf_pt_*_trace`).
- Interpret **device** effects (TensorFlow on GPU vs. CPU) when judging “parity.”

---

## 2. Background: EQCCT and why framework parity matters

### 2.1 EQCCT in brief

EQCCT uses a **convolutional front end**, **patch tokenization**, **transformer** blocks, and a **Conv1D** picking head with sigmoid outputs for P or S probability traces (Saad et al., 2023). Training is performed in **TensorFlow / Keras**. Two checkpoints are typical: one for the **P** branch and one for the **S** branch (different depth and head).

### 2.2 Connection to SeisBench and EQCCTPro

- **SeisBench** provides standardized datasets and a **PyTorch** model interface for many pickers; integrating EQCCT as a **PyTorch** module allows `from_pretrained`-style sharing and apples-to-apples comparisons with PhaseNet-style models.
- **EQCCTPro** (RAPID) benchmarks real-time orchestration across multiple pickers; **EQCCT** is implemented with **TensorFlow** in that stack. TensorFlow’s **per-task** initialization and **XLA** behavior on GPU can dominate wall time under **Ripper**-style parallelism, whereas **Model-Actor** amortizes load cost—*provided* the loaded model is numerically trustworthy on the chosen device (see EQCCTPro draft, §3.2–3.3). Weight transfer that preserves **bit-level or near–bit-level** agreement with the original trainer reduces doubt when moving from research checkpoints to production actors.

---

## 3. Methods: transferring weights from TensorFlow to PyTorch

### 3.1 Canonical PyTorch architecture

Porting succeeds only if **one** PyTorch module hierarchy is designated as canonical. In our codebase, **`EQCCTModelP`** and **`EQCCTModelS`** in `predictor_pt_p` match the **named** Keras layers expected by transfer scripts: multi-head attention uses **`attn.o`** (output projection), transformer MLP uses **`mlp.fc1` / `mlp.fc2`**, and the **S** model exposes **`extra_pre` / `extra_post`** conv stacks consistent with `create_cct_modelS`.

**Lesson:** Mixing alternate ports (`predictor_pt` vs. `predictor_pt_p`, different `state_dict` keys) produces silent partial loads.

### 3.2 Flattening Keras checkpoints

Keras `.h5` weights are walked into a **flat** dictionary: keys like `patch_encoder/dense/kernel:0`, `picker_P/kernel:0`, `multi_head_attention/query/kernel:0`. A small normalization step collapses duplicated path segments (e.g., `conv1d/conv1d/kernel:0` → `conv1d/kernel:0`) so **ordering** matches catalog inspection.

### 3.3 Pairing tensors to PyTorch parameters

Weights are **not** copied by name alone in one shot: **layout transforms** are required.


| Keras concept                  | PyTorch `nn`                     | Typical transform                              |
| ------------------------------ | -------------------------------- | ---------------------------------------------- |
| `Dense` kernel `(in, out)`     | `Linear.weight` `(out, in)`      | **Transpose**                                  |
| `Conv1D` `(k, cin, cout)`      | `Conv1d.weight` `(cout, cin, k)` | **Permute** axes                               |
| `BatchNorm` moving mean/var    | `BatchNorm1d` buffers            | Direct copy after key alignment                |
| `MultiHeadAttention` Q/K/V/out | Split `Linear` projections       | **Reshape** 3D kernels to 2D per project rules |


**Critical bug (P model):** `_coerce_linear` must **prefer `W.T`** when both `W` and `W.T` match a square target (e.g., **40×40** MLP blocks). Keras stores `(in, out)`; PyTorch expects `(out, in)`. For square matrices, a naive “shape match → no transpose” leaves weights **wrong** while still passing crude checks on a few layers.

**Critical bug (S model):** The loader must copy **`picker_S`** (`kernel` + `bias`) into **`model.head.conv`**. Omitting the head leaves random PyTorch initialization at the last layer even if the backbone matches.

### 3.4 Loading weights in TensorFlow for reference parity

`model.load_weights(path)` without **`by_name=True`** can assign tensors in **topological** order. If the **saved file order** does not match the **constructed graph**, middle layers (BN, attention) may receive **wrong** tensors; early convolutions and the picker can still match spot checks while outputs collapse.

**Mitigation:** `load_weights(..., by_name=True, skip_mismatch=False)` when compatible; a **fallback** chain (strict → skip mismatch → positional) with explicit logging when strict loading fails (e.g., Keras 3 vs. older checkpoint layouts for MHA).

### 3.5 PyTorch–TensorFlow implementation alignment

- **MLP activations:** Keras applies **GELU after both** dense layers when `transformer_units` has two entries; the PyTorch `TransformerMLP` must match.
- **Inference mode:** Parity scripts use **`training=False`** and PyTorch **`eval()`** so dropout does not skew comparisons.

### 3.6 Validation pipeline

1. **`parity_p_model`** (and related S checks): Random input, compare TF vs. PT outputs within tolerance; optional weight-vs-disk checks for conv and picker layers.
2. **`tf_pt_p_trace` / `tf_pt_s_trace`:** Per-parameter comparison after the same coercion as the loader; **per-stage activation** comparison (post-front, post-encoder, per transformer block, pre-picker).
3. **`--block0-substeps`:** Finer checkpoints inside the first transformer block to localize the first diverging op (we found MLP before fixing square dense transpose).
4. **SeisBench windows:** Real traces (e.g., STEAD/TXED), 6000 samples, ZNE, normalized per window; overlay TF vs. PT probability curves and catalog picks.
5. **Optional:** TF → ONNX → ONNX Runtime to confirm the **exported** TF graph is self-consistent; PT vs. ORT mismatch then isolates the **hand-written** PyTorch graph.

### 3.7 Device effects (TensorFlow GPU vs. CPU)

TensorFlow on **GPU** can diverge from CPU reference due to **XLA**, **cuDNN**, and **PTX assembler** version warnings (e.g., `ptxas` < 11.1). For **reference parity**, force TensorFlow to **CPU** in validation scripts; for **deployment** studies, compare CPU vs. GPU explicitly. PyTorch may use CUDA independently; that does not imply TensorFlow sees GPUs if the **Jupyter kernel** points at a different Python environment than the training conda env.

---

## 4. Suggested figures (3–4)

**Figure 1.** Schematic: Keras HDF5 → flat dict → layout coercion → PyTorch `EQCCTModelP` / `EQCCTModelS` → SeisBench / Lightning hook-up. Optional side-by-side **P** vs. **S** branch depth (9 vs. 33 conv groups).

**Figure 2.** Validation funnel: random-input max error; layer-trace first failure (e.g., block 0 MLP); end-to-end |TF − PT| distribution over seeds.

**Figure 3.** Example **6000-sample** ZNE window with catalog P/S; overlay **P** and/or **S** probability traces for TF (CPU) and PT.

**Figure 4. (optional)** Operational angle: excerpt from EQCCTPro-style timing table or cartoon of **Ripper** vs. **Model-Actor** with a single line stating that **verified** PyTorch weights enable the same numerical picker inside persistent workers without retraining.

*[Figure files are placeholders; generate from `EQCCT_to_Seisbench/notebooks/tf_pt_working` and validation scripts.]*

---

## 5. Discussion

### 5.1 Why this matters for seismic networks

Networks that invest in **TensorFlow** training should not be locked out of **PyTorch-first** tooling. A documented transfer path:

- Reduces **duplicate training** and carbon cost.
- Increases **trust** (numerical parity vs. original checkpoints).
- Simplifies **publication** of weights alongside open code.

### 5.2 Limitations

- **Version lock-in:** TensorFlow, Keras, CUDA, and driver stacks must be documented next to each released `.pt` / `.h5` pair.
- **Jupyter vs. CLI:** Kernels must match the conda environment used for validation; otherwise TF may list **no GPUs** or load different package builds.
- **Not every** custom layer has a one-line map; attention **bias** shapes and Keras 3 **MHA** naming may require loader updates when upgrading TensorFlow.

### 5.3 Relation to EQCCTPro / RAPID

The EQCCTPro draft emphasizes that **orchestration overhead** dominates many real-time scenarios. Weight transfer does not remove TensorFlow initialization cost in a **TensorFlow** worker, but it **does** enable an alternative **PyTorch** worker with identical picks—useful where PyTorch deployment is preferred. The same validation discipline (CPU reference, careful GPU comparison) applies whenever **XLA** warnings appear.

---

## 6. Data, testing environment, and code availability

### 6.1 Seismic data and EQCCT model provenance

Following the **Data and Resources** statement in the companion EQCCTPro manuscript (parallel RAPID / real-time orchestration study), **seismic waveform data** and **computational hardware** for network-scale experiments were provided by the **Texas Seismological Network** and **Seismology Research Team (TexNet)**. Continuous and event waveforms used in that work were obtained from TexNet’s **FDSN** service and are publicly available at `http://rtserve.beg.utexas.edu/` (see Savvaidis et al., 2019).

**EQCCT** is an open-source deep-learning model (**Saad et al., 2023**). Official TensorFlow training artifacts and architecture definitions are distributed with the EQCCT project; repository locations cited in the EQCCTPro draft include the main tree (`https://github.com/ut-beg-texnet/eqcct/tree/main`), **EQCCTOne**, and **EQCCTPro** (`…/eqcct/tree/main/eqcctpro`). The **Keras** checkpoints used for weight transfer in this study are the **same class of P/S branch weights** associated with that published EQCCT lineage (institutional TexNet training and author-provided models, as in the operational literature: **Chen et al., 2024**, *Earth and Space Science*).

For **SeisBench**-style qualitative validation (STEAD, TXED; 100 Hz, three-component windows), we use public benchmark datasets distributed through **SeisBench** (**Woollam et al., 2022**; dataset studies such as **Münchmeyer et al., 2022**). These are **not** TexNet proprietary holdings; they serve to demonstrate parity on diverse community waveforms after numerical checks against TensorFlow.

### 6.2 In-house testing (same system as EQCCTPro benchmarks)

Numerical parity scripts (`parity_p_model`, `tf_pt_p_trace`, `tf_pt_s_trace`), ONNX cross-checks where applicable, and notebook demonstrations (**TF vs. PyTorch** probability traces, optional **CPU vs. GPU** TensorFlow comparisons) were run **in house** on the **same class of workstation** employed for the EQCCTPro / RAPID timing and memory study: **dual AMD Ryzen Threadripper PRO 7985WX** CPUs (128 logical cores total), **512 GB** DDR5 RAM, and **two NVIDIA RTX 6000 Ada Generation** GPUs (49 GB VRAM each), with **CPU/GPU affinity** controlled for reproducibility as described in **§2.3** of `EQCCTPro_Draft.md`. Thus, wall-time and device-behavior statements about TensorFlow **XLA**, **cuDNN**, and **ptxas** warnings are comparable to the EQCCTPro paper’s experimental environment.

### 6.3 Code and checkpoint availability

- **TensorFlow ↔ PyTorch port and validation:** `EQCCT_to_Seisbench` (implementation package `eqcct_sb`, notebooks under `notebooks/tf_pt_working`, validation under `eqcct_sb/validation`). *[Insert public repository URL, branch, and version tag or commit hash at release.]*
- **Reference weights:** Keras **`.h5`** checkpoints for `modelP` / `modelS`; optional exported **PyTorch** **`.pt`** files from `python -m eqcct_sb.validation.tf_pt_p_trace` / `tf_pt_s_trace` with `--save-model`.
- **Environment diagnostics:** `python -m eqcct_sb.validation.tf_gpu_env_sniff` (TensorFlow GPU visibility vs. Jupyter kernel).

### 6.4 Acknowledgements

The authors thank **TexNet** and the **State of Texas** for support of related operational and software work (see acknowledgement language in **EQCCTPro_Draft.md**, including University of Texas at Austin award **#201503664**, and contributors named there). This weight-transfer note is a **methods** companion to the EQCCTPro orchestration manuscript; citations above align with that draft’s bibliography.

---

## 7. Conclusion

We outlined a **reproducible, validated** path to bring **TensorFlow-trained EQCCT** weights into **PyTorch** for **SeisBench**-compatible distribution. The central technical lessons—**strict Keras load semantics**, **correct dense transforms for square layers**, **complete head loading for S**, and **automated** parity tooling—generalize to other institutional pickers. We invite seismic software venues to treat such **transfer manuals** as first-class contributions alongside new architectures.

---

## References

References below are aligned with **`EQCCTPro_Draft.md`** (eqcctpro/docs) so this methods note shares a consistent bibliography with the RAPID / EQCCTPro real-time parallelization paper. Format for the target journal (author–year vs. numbered) should be applied at submission.

Ali, M. (2023, Jul). Distributed processing using ray framework in python.

Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. In *Proceedings of the April 18-20, 1967, Spring Joint Computer Conference, AFIPS ’67 (Spring)*, New York, NY, USA, pp. 483–485. Association for Computing Machinery.

Bates, D. and D. Watts (2008, 05). *Nonlinear Regression Analysis and Its Applications*, pp. 32–66.

Chen, Y., O. M. Saad, A. Savvaidis, F. Zhang, Y. Chen, D. Huang, H. Li, and F. Aziz Zanjani (2024). Deep learning for p-wave first-motion polarity determination and its application in focal mechanism inversion. *IEEE Transactions on Geoscience and Remote Sensing* **62**, 1–11.

Chen, Y., Savvaidis, A., Siervo, D., Huang, D., & Saad, O. M. (2024). Near real-time earthquake monitoring in Texas using the highly precise deep learning phase picker. *Earth and Space Science*, **11**, e2024EA003890. https://doi.org/10.1029/2024EA003890

Feng, T., S. Mohanna, and L. Meng (2022). Edgephase: A deep learning model for multi-station seismic phase picking. *Geochemistry, Geophysics, Geosystems* **23**(11), e2022GC010453.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics* **29**(5), 1189–1232.

Gustafson, J. L. (1988, May). Reevaluating amdahl’s law. *Commun. ACM* **31**(5), 532–533.

Herlihy, M. and N. Shavit (2012). *The Art of Multiprocessor Programming, Revised Reprint*. Morgan Kaufmann.

Johnston, J. and J. DiNardo (1997). *Econometric Methods* (4th ed.). McGraw-Hill.

Lim, C. S. Y., S. Lapins, M. Segou, and M. J. Werner (2025). Deep learning phase pickers: how well can existing models detect hydraulic-fracturing induced microseismicity from a borehole array? *Geophysical Journal International* **240**(1), 535–549, https://doi.org/10.1093/gji/ggae386

McCool, M., J. Reinders, and A. Robison (2012). *Structured Parallel Programming: Patterns for Efficient Computation*. ITPro collection. Elsevier Science.

Moritz, P., R. Nishihara, S. Wang, A. Tumanov, R. Liaw, X. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan, and I. Stoica (2018). Ray: A Distributed Framework for Emerging AI Applications. In *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*, Carlsbad, CA, pp. 561–577. USENIX Association.

Mousavi, S., W. Ellsworth, Z. Weiqiang, L. Chuang, and G. Beroza (2020, 08). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. *Nature Communications* **11**, 3952.

Mousavi, S. M. and G. C. Beroza (2022). Deep-learning seismology. *Science* **377**(6607), eabm4470.

Münchmeyer, J., J. Woollam, A. Rietbrock, F. Tilmann, D. Lange, T. Bornstein, T. Diehl, C. Giunchi, F. Haslinger, D. Jozinović, A. Michelini, J. Saul, and H. Soto (2022, 01). Which picker fits my data? a quantitative evaluation of deep learning based seismic pickers. *Journal of Geophysical Research: Solid Earth* **127**.

Ray.io (n.d.). What is ray core? https://docs.ray.io/en/latest/ray-core/walkthrough.html. Accessed: 2023-11-13.

Saad, O. M. and Y. Chen (2022). Capsphase: Capsule neural network for seismic phase classification and picking. *IEEE Transactions on Geoscience and Remote Sensing* **60**, 1–11.

Saad, O. M., Y. Chen, A. Savvaidis, W. Chen, F. Zhang, and Y. Chen (2022). Unsupervised deep learning for single-channel earthquake data denoising and its applications in event detection and fully automatic location. *IEEE Transactions on Geoscience and Remote Sensing* **60**, 1–10.

Saad, O. M., Y. Chen, A. Savvaidis, S. Fomel, and Y. Chen (2022). Real-time earthquake detection and magnitude estimation using vision transformer. *Journal of Geophysical Research: Solid Earth* **127**(5), e2021JB023657.

Saad, O. M., Y. Chen, D. Siervo, F. Zhang, A. Savvaidis, G.-c. D. Huang, N. Igonin, S. Fomel, and Y. Chen (2023). Eqcct: A production-ready earthquake detection and phase-picking method using the compact convolutional transformer. *IEEE Transactions on Geoscience and Remote Sensing* **61**, 1–15.

Saad, O. M., G. Huang, Y. Chen, A. Savvaidis, S. Fomel, N. Pham, and Y. Chen (2021). Scalodeep: A highly generalized deep learning framework for real-time earthquake detection. *Journal of Geophysical Research: Solid Earth* **126**(4), e2020JB021473.

Savvaidis, A., B. Young, G. D. Huang, and A. Lomax (2019, 06). Texnet: A statewide seismological network in texas. *Seismological Research Letters* **90**(4), 1702–1715.

Sheen, D.-H., Y.-J. Seong, S. Jung, and J.-h. Park (2023). Real-time application of deep learning seismic phase picker to earthquake monitoring. Poster S31G-0424, *AGU Fall Meeting*, San Francisco, CA, 11–15 December, session *Machine Learning–Driven Analysis of Geophysical Signals III*.

Tan, Y. J., F. Waldhauser, W. Ellsworth, M. Zhang, Z. Weiqiang, M. Michele, L. Chiaraluce, G. Beroza, and M. Segou (2021, 04). Machine-learning-based high-resolution earthquake catalog reveals how complex fault structures were activated during the 2016–2017 central italy sequence. *The Seismic Record* **1**, 11–19.

Wang, T., D. Trugman, and Y. Lin (2021). Seismogen: Seismic waveform synthesis using gan with application to seismic data augmentation. *Journal of Geophysical Research: Solid Earth* **126**(4), e2020JB020077.

Woollam, J., J. Münchmeyer, F. Tilmann, A. Rietbrock, D. Lange, T. Bornstein, T. Diehl, C. Giunchi, F. Haslinger, and D. Jozinović (2022). SeisBench—A Toolbox for Machine Learning in Seismology. *Seismological Research Letters* **93**(3), 1695–1709.

Xiao, Z., J. Wang, C. Liu, J. Li, L. Zhao, and Z. Yao (2021). Siamese earthquake transformer: A pair-input deep-learning model for earthquake detection and phase picking on a seismic arrival-time picking method. *Journal of Geophysical Research: Solid Earth* **126**(5), e2020JB021444.

Yeck, W. L., J. M. Patton, Z. E. Ross, G. P. Hayes, M. R. Guy, N. B. Ambruz, D. R. Shelly, H. M. Benz, and P. S. Earle (2020). Leveraging Deep Learning in Global 24/7 Real-Time Earthquake Monitoring at the National Earthquake Information Center, *Seismol. Res. Lett.* **92**, 469–480, doi: 10.1785/0220200178.

Yu, Z., Y. Jiang, X. Jing, and H. Zheng (2024). Study on geomagnetic observations associated with three major earthquakes in southwest china through a novel deep learning framework. *IEEE Transactions on Geoscience and Remote Sensing* **62**, 1–18.

Zhu, W. and G. C. Beroza (2018, 10). Phasenet: a deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International* **216**(1), 261–273.

---

## Appendix A: Suggested checklist for other TF models

1. Freeze **one** PyTorch `nn.Module` as canonical; delete or quarantine alternate ports.
2. Implement **`flat_h5`** + **`_grab`** with regex fallbacks for renamed layers.
3. Audit **Dense** and **Conv1d** coercion; add **unit tests** for square matrices.
4. Require **`load_weights(by_name=True)`** for Keras reference.
5. Ship **`tf_pt_trace`**-style tooling for any new branch (depth differs from P).
6. Document **conda** env and **`tensorflow[and-cuda]`** vs. CPU wheels for GPU notebooks.

---

## Appendix B: Building the EQCCTPro PDF (project context)

The EQCCTPro repository includes `docs/build_pdf.sh`, which runs **pandoc** and **tectonic** on `EQCCTPro_Draft.md` to produce `EQCCTPro_Draft.pdf`. You can adapt the same pipeline for this manuscript once figures and bibliography are finalized.

```bash
# From eqcctpro project root (see docs/build_pdf.sh)
bash docs/build_pdf.sh
```

---

*Draft generated to support submission as a short methods / software paper. Revise section numbering, author list, and journal-specific formatting before submission.*