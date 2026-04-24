# Porting institutional deep-learning pickers to PyTorch and SeisBench: a practical guide to TensorFlow–PyTorch weight transfer (EQCCT case study)

**Authors:** C. Skevofilax, A. Savvaidis

**Affiliations:** The University of Texas at Austin

**Keywords:** seismic phase picking; deep learning; model portability; TensorFlow; PyTorch; SeisBench; EQCCT; weight transfer; reproducibility

---

## Abstract

Seismic research groups often train their own phase-picking models using whatever machine learning framework fits best into their existing workflow, and TensorFlow/Keras is still a popular choice. Meanwhile, much of the modern open-source seismology ecosystem has standardized on PyTorch. Tools like SeisBench, which are widely used in the community, are built entirely in PyTorch. Recent surveys show that PyTorch now dominates deep learning research, and many toolkits have adopted it to make their software easier to use and more accessible.

This creates a practical problem: models trained in TensorFlow cannot be added to platforms like SeisBench without first being converted to PyTorch. Without a reliable way to convert models between frameworks, research groups cannot easily share their work or integrate it into modern workflows, which limits both usability and long-term impact.

In this work, we present a practical, step-by-step workflow for converting transformer-based phase pickers, specifically EQCCT, from TensorFlow to PyTorch. We use our open-source implementation as a case study and describe the core challenges that determine whether conversion is scientifically and operationally trustworthy: weight-loading inconsistencies in Keras, ambiguity when mapping dense layers with square weight matrices, missing output branches in earlier model versions, and device-dependent numerical differences on GPU. We validate the converted model with a staged protocol that combines layer-wise weight and activation checks with end-to-end comparisons on real waveforms, including a SeisBench-scale benchmark on 100,000 windows from TXED and STEAD. In that benchmark, mean absolute error between TensorFlow and PyTorch probability traces is \( \sim 10^{-9} \) on CPU and \(10^{-7}\)–\(10^{-6}\) on GPU, with all windows remaining below \(10^{-2}\) in worst-case pointwise discrepancy. The result is a reproducible guide for making non-PyTorch pickers compatible with PyTorch-centric tooling without retraining, and a template for reporting numerical parity in a form that supports deployment in persistent model-worker systems such as RAPID.

## 1. Introduction

### 1.1 Motivation

Deep learning phase picking models have rapidly emerged in seismology over the past decade. Models like PhaseNet (Zhu & Beroza, 2019), EQTransformer (Mousavi et al., 2020), and EQCCT (Saad et al., 2023) have significantly improved seismic event detection and catalog automation, to the point where they are now integrated into real-time production workflows (Skevofilax et al., 2026, Muñoz et al., 2025). By standardizing these models into a Python-based toolkit, SeisBench (Woollam et al., 2022) has lowered the barrier between research and application, fostering a community where researchers can more easily share their work with a broader audience.

Like many research institutions and open-source projects, SeisBench has adopted PyTorch (PT, Paszke et al., 2019) as its standard machine learning framework. This adoption reflects a broader trend in the research community, where PyTorch now accounts for approximately 85% of deep learning papers (SecondTalent, 2026). The choice helps mitigate inconsistencies in model functionality across different frameworks. By standardizing on a single architecture, SeisBench avoids the complex code handling typically required when supporting multiple frameworks, creating a more predictable and unified workflow.

However, seismic networks and research groups like the Texas Seismological Network (TexNet, Savvaidis et al., 2019) already have trained models using alternatives to PyTorch, such as TensorFlow (TF, Abadi et al., 2015) with their EQCCT model. Because of SeisBench's PyTorch-based approach, groups like TexNet cannot easily integrate their models into the toolkit without TensorFlow-based accessors. This limits the potential for advancing seismic research and real-world applications. A critical question therefore emerges: How can a network deliver an EQCCT-class model to the community in a PyTorch-compatible form without discarding years of carefully tuned weights?

While resources exist for transferring model weights between frameworks (Czyzewski et al. 2021), available tools and tutorials are limited, particularly for addressing domain-specific challenges. These include maintaining prediction accuracy, managing inference time and memory consumption, and preserving adversarial robustness when converting trained models across different frameworks and deployment environments (Wang et al. 2022). We encountered many of these issues ourselves when transferring EQCCT from TensorFlow to PyTorch. In this work, we provide a practical guide to help future users more easily navigate the weight transfer process.

In this work, we provide a practical, reproducible guide for transferring transformer-based phase picking models from TensorFlow to PyTorch, using EQCCT as a case study. We chose EQCCT because it is actively used in production by a large-scale seismic network (TexNet), has been sought for integration into SeisBench for several years, and its dual-branch architecture (separate P and S models with different convolutional structures) presents additional complexity that tests the robustness of the conversion process.

Our contribution is methodological rather than architectural. We demonstrate how to map Keras HDF5 checkpoints to PyTorch state dictionaries, establish numerical parity between framework implementations through layer-wise validation, and verify model fidelity using real waveform data in SeisBench-compatible formats. We also address practical challenges including device-specific numerical differences and weight-loading ambiguities that arise during conversion.

## 2. Background: EQCCT and why framework parity matters

### 2.1 EQCCT in brief

EQCCT consists of a convolutional front end, patch tokenization, several transformer blocks, and a Conv1D picking head with sigmoid outputs that produce probability traces for P or S phases (Saad et al., 2023). The model was trained in TensorFlow/Keras and typically uses two separate checkpoints: one for the P branch and one for the S branch, which differ in depth and architecture.

### 2.2 SeisBench, production deployment, and numerical parity

SeisBench provides standardized datasets and a PyTorch-oriented model interface that allows researchers to load many published pickers in a uniform way, including pretrained weights where authors have supplied them (Woollam et al., 2022). Bringing EQCCT into that ecosystem as a native PyTorch module matters because it lowers the barrier for side-by-side comparison with other community models and for adoption by groups that already standardize on SeisBench for benchmarking and teaching. This is not merely convenience, as when weights are ported rather than retrained, users need confidence that the ported model is the same function as the original trainer up to well-understood numerical tolerances.

The same question arises outside the research stack. EQCCT is also used in production-oriented settings, including orchestration through the RAPID framework (Skevofilax et al., 2026) and application with SCMLPick (Muñoz et al., 2025), where TensorFlow’s per-task startup costs and GPU-oriented compilation paths can figure prominently in wall time when models are loaded repeatedly across parallel tasks. Architectures that keep a persistent model worker can amortize load cost, but that strategy only makes sense if the weights loaded into the worker reproduce the training-time behavior closely enough for operational decisions. Numerical parity between TensorFlow and PyTorch is therefore tied both to reproducibility in publications and to reliability when the same picks are expected in live pipelines.

Despite the clear need for reliable framework conversion, practical guidance remains limited. While general frameworks for model conversion exist, including ONNX-based approaches, and some tutorials address specific conversion scenarios, comprehensive documentation of the challenges and validation requirements for domain-specific models like seismic phase pickers is scarce. Empirical studies have shown that framework conversion introduces challenges including prediction accuracy degradation, performance variations, and numerical inconsistencies (Wang et al., 2022), but systematic workflows for addressing these issues in production seismology contexts remain underdeveloped. In the following section, we present a systematic workflow for converting EQCCT from TensorFlow to PyTorch while maintaining numerical fidelity, along with the validation strategy needed to verify that the ported model faithfully reproduces the original.

### 2.3 Challenges in framework conversion (why parity is nontrivial)

Framework conversion fails most often not because the model architecture is misunderstood, but because implementation details that look interchangeable are not. First, multiple layers require tensor layout changes (e.g., Dense kernels and Conv1D kernels), and those changes can silently become ambiguous when weight matrices are square. Second, modern attention layers store parameters in shapes that differ across APIs, and name-based loading can break across Keras versions even when the checkpoint is intact. Third, GPU execution introduces legitimate numerical divergence due to kernel selection and compiler fusion, so “the same weights” does not automatically imply bitwise-identical outputs. These factors motivate a validation strategy that makes parity a measured claim—reported with explicit thresholds and device profiles—rather than an assumption.

## 3. Methods

### 3.1 Conversion workflow overview

The conversion workflow proceeds in three stages. First, we establish a single canonical PyTorch implementation for each model branch to ensure every layer has a clear target structure. Second, we extract weights from the Keras HDF5 checkpoints and transform them to match PyTorch's expected tensor layouts. Third, we validate the converted model against the TensorFlow original, starting with controlled tests on random inputs and progressing to real seismic waveforms from SeisBench datasets. The subsections below follow this progression and highlight the main challenges we encountered at each stage.

### 3.2 Architectural alignment between frameworks

Successful porting requires designating a single canonical PyTorch module hierarchy for each model branch. In our implementation, the P- and S-branch models defined in predictor_pt_p serve as the canonical EQCCTModelP and EQCCTModelS classes. Their state_dict keys follow naming conventions that align with our transfer scripts: multi-head attention output projections appear as attn.o, MLP layers as mlp.fc1 and mlp.fc2, and the S-branch convolution stacks as extra_pre and extra_post, matching the structure of create_cct_modelS in the TensorFlow reference. This consistency is critical because alternative implementations with different key layouts can silently load only a subset of weights without triggering errors, which degrades model outputs in ways that may not be immediately obvious as discussed later.

Beyond naming conventions, implementation details must match the Keras computational graph. For instance, when transformer_units specifies two equal projection dimensions, Keras applies GELU activation after both dense layers in the MLP block. The PyTorch TransformerMLP must replicate this exact ordering. Similarly, for all parity experiments we run TensorFlow with training=False and set PyTorch to evaluation mode so that dropout and batch normalization behavior align with inference settings.

### 3.3 Weight extraction and transformation

Keras stores weights in HDF5 files using a nested hierarchical structure. We flatten each file into a dictionary with keys that resemble filesystem paths, such as patch_encoder/dense/kernel:0 and multi_head_attention/query/kernel:0. A normalization step then collapses duplicated path segments that sometimes appear in Keras exports, for example converting conv1d/conv1d/kernel:0 to conv1d/kernel:0, so that tensor ordering remains consistent with both manual inspection and automated pairing logic.


Table 1 summarizes the main structural differences between Keras tensors and their PyTorch counterparts. These transformations must be applied systematically. Without careful attention to layout, middle layers can end up with incorrect weights even while early layers pass spot checks.

| Keras concept | PyTorch module | Typical transform |
| --- | --- | --- |
| Dense kernel, shape (in, out) | Linear.weight, shape (out, in) | Transpose |
| Conv1D kernel (k, cin, cout) | Conv1d.weight (cout, cin, k) | Permute axes |
| BatchNorm moving statistics | BatchNorm1d buffers | Copy after key alignment |
| MultiHeadAttention Q, K, V, output | Separate Linear projections | Reshape 3D Keras kernels to 2D per projection rules |

A recurring challenge arose when mapping dense layers in the P-branch MLP blocks. Keras stores dense kernels with shape (in, out), while PyTorch Linear layers expect (out, in). Our conversion routine must apply the transpose even when both the original matrix and its transpose have the same shape. For square matrices—such as the 40×40 blocks in the MLP—a naive implementation that accepts the first shape match without transposing will leave weights incorrect while still passing superficial checks on unrelated layers. Enforcing the transpose rule for all dense layers eliminated a persistent activation mismatch we observed in the first transformer block.

A second challenge specific to the S branch involves completeness. The loader must explicitly copy the picker_S kernel and bias into the final convolutional head of the PyTorch model. If only the backbone loads successfully, the output layer remains randomly initialized even though earlier layers appear correct—a problem that can go unnoticed until end-to-end validation.

Loading order also matters when establishing reference comparisons in TensorFlow. Calling load_weights without name-based assignment binds tensors in graph order. If the checkpoint's internal ordering does not match the constructed model, batch normalization and attention layers may receive incorrect weights while early convolutions still load correctly. We therefore use strict name-based loading whenever the checkpoint format supports it, falling back through a sequence of strategies (strict, then skip-mismatch for known Keras 3 layout differences, then positional) with explicit logging when strict loading fails.

To improve reproducibility for readers applying the same approach, Algorithm 1 summarizes the checkpoint flattening and key normalization step, and Algorithm 2 summarizes the dense-layer coercion rule that avoids the square-matrix ambiguity described above.

**Algorithm 1 (HDF5 flatten + path normalization).** Walk the HDF5 file recursively and emit a flat dict from normalized key to NumPy array.

1. For each dataset at path `p` in the HDF5 tree, read it as an array `A`.
2. Normalize `p` by collapsing duplicated segments (e.g., `conv1d/conv1d/kernel:0 → conv1d/kernel:0`).
3. Insert `out[p] = A`.

**Algorithm 2 (Dense kernel coercion with square-matrix safeguard).** Map a Keras Dense kernel `W` to a PyTorch Linear weight tensor with target shape `tgt = (out, in)`.

1. If `W.T.shape == tgt`, use `W.T` (preferred, even when square).
2. Else if `W.shape == tgt`, use `W`.
3. Else fail with a shape mismatch error and report the candidate shapes.

### 3.4 Validation strategy

We validated the conversion in stages so that any failures could be traced to a specific part of the model before committing to large-scale waveform tests.

We began with random-input checks that compare TensorFlow and PyTorch outputs across the entire forward pass, verifying agreement within numerical tolerances. These checks optionally include direct comparisons between weights stored on disk and the loaded tensors for convolutional and picker layers. The trace utilities tf_pt_p_trace and tf_pt_s_trace extend this approach by comparing parameters after applying the same transformation rules used in the production loader, and by comparing activations at key stages: after the convolutional front end, after the patch encoder, within each transformer block, and before the picker head. An optional block0-substeps mode adds finer checkpoints inside the first transformer block, which we used to pinpoint the first diverging operation when the MLP mapping was still incorrect.

We then ran a dataset-scale benchmark (eqcct_sb.validation.tf_pt_seisbench_dataset_benchmark) on community waveforms exposed through SeisBench. That program iterates metadata from the TXED and STEAD benchmark collections (Woollam et al., 2022; Münchmeyer et al., 2022), resampled to 100 Hz with component order ZNE, and retains only traces that have both catalog P and S sample indices and are long enough to extract a 6000-sample window centered on P with S falling inside the window. For each qualifying row we build one window, apply the same per-window time normalization used in our notebook parity demos, and run the P and S TensorFlow and PyTorch models on identical NumPy inputs. In the experiments reported below we set a cap of 50 000 such windows per dataset with stride 1 along the filtered metadata, which yields up to 100 000 windows total when both corpora are traversed sequentially (50 000 from TXED followed by 50 000 from STEAD). The full public releases of TXED and STEAD contain more traces than this cap; our setting is therefore a large fixed subsample chosen for computational cost, not an exhaustive enumeration of every row in either catalog.

For each window we compute pointwise differences between the sigmoid probability traces from TensorFlow and PyTorch for the P branch and separately for the S branch. Aggregates are reported with online (Welford) statistics over windows: mean squared error (MSE) and mean absolute error (MAE) between the two frameworks, computed over all time samples and components of the probability output; the mean and standard deviation of the per-window maximum absolute discrepancy; the global maximum absolute discrepancy observed on any window; and the fraction of windows whose per-window maximum absolute discrepancy falls below fixed thresholds (10⁻⁶ through 10⁻²). These metrics measure numerical agreement between implementations, not picking accuracy against a human or catalog pick. We repeat the benchmark under two execution profiles: TensorFlow and PyTorch both on CPU (TensorFlow reference on /CPU:0, PyTorch on the CPU device) and both on GPU (TensorFlow on /GPU:0, PyTorch on cuda:0), so that we can separate weight-transfer fidelity from device-specific numerical behavior.

To make the staged validation actionable, Table 2a summarizes the validation gates we use, what they test, and what constitutes an informative failure. The SeisBench-scale benchmark is the final gate: it does not replace layer-wise traces, but it ensures that any residual discrepancies remain small across diverse real waveforms and across device profiles.

**Table 2a.** Validation gates and pass/fail criteria used in this study.

| Gate | What it tests | Output | Typical failure signal |
| --- | --- | --- | --- |
| Random-input parity | End-to-end forward equivalence under controlled inputs | Max and mean \|TF−PT\| on outputs | Large mismatch indicates layout or missing weights |
| Weight trace | Correct parameter mapping (after coercion) | Per-layer allclose checks with rtol/atol | Localizes wrong transpose/permute |
| Activation trace | Correct intermediate semantics (per stage) | Stage-wise max/mean \|Δ\| | Pinpoints first diverging block/op |
| SeisBench windows | End-to-end behavior on real waveforms | MSE/MAE, max \|Δ\|, fractions below thresholds | Surfaces device effects and rare worst cases |

Where useful, we also exported the TensorFlow graph to ONNX and compared it against ONNX Runtime to confirm internal consistency. Disagreement between PyTorch and ONNX Runtime in that setup isolates issues in the hand-written PyTorch implementation rather than in the TensorFlow export process.

Figure 2 illustrates the activation gate quantitatively. For both the P and S branches, and for each named architectural checkpoint (convolutional front end → patch encoder → each transformer block → final layer normalization → pre-picker reshape), we measure mean and maximum |TF − PT| activations on five random ZNE inputs and repeat the measurement on CPU and on the first GPU. The CPU profile produces stage-wise discrepancies near floating-point round-off and grows only weakly with depth, which is consistent with correct weight loading and matched reference kernels. The GPU profile shows the expected uniform offset—roughly two orders of magnitude larger but still bounded—reflecting kernel choice and fusion in cuDNN/XLA-style paths rather than any new error introduced by the conversion. The fact that both curves remain monotonic and well-bounded along the architecture is what makes the SeisBench-scale benchmark in Section 4 trustworthy as an end-to-end test rather than an integral over uncontrolled error sources.

**Figure 2.** Layer-by-layer activation discrepancy between TensorFlow and PyTorch at each architectural checkpoint of the EQCCT P branch (left column) and S branch (right column). Top row: maximum |TF − PT| within the activation tensor at each stage. Bottom row: mean |TF − PT|. Bars are means over five random ZNE inputs; whiskers show the across-input standard deviation. The CPU profile (darker shade) sits near floating-point precision; the GPU profile (lighter shade) is uniformly larger but remains bounded across all stages. Source data: results/layer_activations.json (run eqcct_sb.validation.tf_pt_layer_activations); plot generated by scripts/plot_layer_activations.py. File: figures/tf_pt_layer_activations.png.

![Figure 2. Per-stage TF vs PT activation discrepancy for the P and S branches under CPU and GPU profiles.](figures/tf_pt_layer_activations.png)

## 4. Results

Table 2 summarizes the large-scale benchmark on 100 000 SeisBench windows (50 000 from TXED and 50 000 from STEAD under the protocol in Section 3.4). All windows completed without I/O or runtime errors; none were skipped for short traces or S falling outside the P-centered window in these runs. On CPU, mean discrepancies between TensorFlow and PyTorch stay below 10⁻⁸ in MAE for both branches, and every window had a per-point maximum absolute difference below 10⁻⁴ relative to the paired TensorFlow output. That profile is the strictest check that the ported graph matches the Keras reference when both frameworks use deterministic CPU kernels.

On GPU, differences are larger but remain modest in operational terms for probability-valued traces: mean MAE stays near 10⁻⁷ for P and 10⁻⁶ for S, and 89.5% (P) and 70.6% (S) of windows still have per-window max |TF − PT| below 10⁻³, with all windows below 10⁻². The elevation relative to CPU is consistent with known variability from cuDNN, XLA-style fusion, and mixed-precision paths on the device stack; it does not indicate a separate bug in the weight map, because the CPU profile exercises the same transferred weights and code paths. Figure 1 contrasts the two branches and the two device profiles using the mean MAE and the mean within-window worst-point error, which are the quantities operators typically care about when asking whether two pickers are “the same” on a fixed window.

**Table 2.** TensorFlow versus PyTorch numerical agreement on 100 000 SeisBench windows (TXED and STEAD, 50 000 per dataset, stride 1). MSE and MAE are averaged over windows; each window contributes one vector of errors over all time points in the P or S probability trace. “Mean max |Δ|” is the mean, over windows, of the maximum absolute difference within that window. “Global max |Δ|” is the worst case over all windows. Fractions are the share of windows with per-window maximum absolute difference below the stated tolerance.

| Quantity | Device | P branch | S branch |
| --- | --- | --- | --- |
| Mean MSE | CPU | 3.4 × 10⁻¹⁶ | 2.2 × 10⁻¹⁶ |
| Mean MSE | GPU | 1.1 × 10⁻¹¹ | 5.8 × 10⁻¹¹ |
| Mean MAE | CPU | 1.4 × 10⁻⁹ | 1.4 × 10⁻⁹ |
| Mean MAE | GPU | 2.3 × 10⁻⁷ | 6.6 × 10⁻⁷ |
| Mean max \|TF − PT\| per window | CPU | 3.2 × 10⁻⁷ | 2.4 × 10⁻⁷ |
| Mean max \|TF − PT\| per window | GPU | 4.7 × 10⁻⁵ | 8.8 × 10⁻⁵ |
| Global max \|TF − PT\| | CPU | 3.6 × 10⁻⁶ | 3.9 × 10⁻⁶ |
| Global max \|TF − PT\| | GPU | 8.8 × 10⁻⁴ | 1.9 × 10⁻³ |
| Share of windows with max \|Δ\| < 10⁻⁴ | CPU | 100% | 100% |
| Share of windows with max \|Δ\| < 10⁻⁴ | GPU | 89.5% | 70.6% |
| Share of windows with max \|Δ\| < 10⁻³ | CPU | 100% | 100% |
| Share of windows with max \|Δ\| < 10⁻³ | GPU | 100% | 99.95% |

**Figure 1.** Comparison of TensorFlow–PyTorch discrepancies on the SeisBench benchmark for the P and S branches under CPU and GPU execution profiles (100 000 windows as in Table 2). Left: mean MAE between paired probability traces. Right: mean over windows of the maximum absolute pointwise difference within each window. Source data: results/tf_pt_benchmark_cpu.json and results/tf_pt_benchmark.json; plot generated by scripts/plot_tf_pt_benchmark_results.py. File: figures/tf_pt_benchmark_tf_pt_discrepancy.png.

![Figure 1. TensorFlow versus PyTorch discrepancy on SeisBench windows (see caption above).](figures/tf_pt_benchmark_tf_pt_discrepancy.png)

Figure 1 reports central tendencies, but two complementary views matter when judging whether converted models are operationally equivalent. The first is the empirical distribution of per-window discrepancies, not just its mean: a low average error is uninformative if a small tail of windows produces outputs that an operator would visibly disagree with. Figure 3 therefore shows the full per-window distributions on a 2 × 2 panel: empirical CDFs of the per-window worst-point error for CPU (panel A) and for the first GPU (panel B), a violin plot of the MAE distribution for both branches and both devices (panel C), and a boxplot of the per-window maximum absolute difference with its outliers (panel D). The CDFs sit close to one well below 10⁻⁴ on both branches and both devices, the violins concentrate near 10⁻⁹ on CPU and 10⁻⁷–10⁻⁶ on GPU, and the boxplots show that even the long tails fall comfortably below the operationally relevant 10⁻² threshold.

**Figure 3.** Per-window error distributions of the TF vs PT comparison on a SeisBench subsample. Panels A and B: empirical CDFs of max |TF − PT| per window on CPU and on GPU, with vertical guides at 10⁻⁶, 10⁻⁴, and 10⁻². Panels C and D: full per-window distributions of MAE (violin plot, log-scale y-axis) and of max |TF − PT| (boxplot with outliers), in each case for the P and S branches under CPU and GPU profiles. Blue = P branch, red = S branch. Source data: results/per_window_errors.npz (run eqcct_sb.validation.tf_pt_per_window_errors); plot generated by scripts/plot_error_distributions.py. File: figures/tf_pt_error_distributions.png.

![Figure 3. CDFs, violins, and boxplots of per-window TF vs PT discrepancies for the P and S branches under CPU and GPU profiles.](figures/tf_pt_error_distributions.png)

The second view that matters is intuitive rather than statistical: do the two probability traces actually look the same on real waveforms? Figure 4 answers this directly on six independently chosen real seismic windows—three drawn at random from TXED and three from STEAD—each with both catalog P and catalog S marked. For each trace we overlay the TensorFlow and PyTorch probability traces of both the P and S branches; the curves are visually indistinguishable, and the inset annotation reports the per-window maximum absolute difference, which sits in the 10⁻⁷–10⁻⁵ range on CPU. This is the qualitative complement to Tables 1–2 and Figure 3: the conversion preserves not just average parity but also pick shapes, peak amplitudes, and arrival times.

**Figure 4.** TensorFlow versus PyTorch P and S probability traces on six real SeisBench windows (three from TXED, three from STEAD), each containing both a catalog P and catalog S inside a 6000-sample window centered on P. Top row: ZNE waveforms (offset for clarity) with vertical dashes at the catalog P (blue) and catalog S (orange) sample positions. Middle and bottom rows: P-branch and S-branch probability traces, with PT shown as a dashed red line on top of the solid blue TF curve. Inset boxes report the per-window max |TF − PT|. Plot generated by eqcct_sb.validation.tf_pt_waveform_compare_figure. File: figures/tf_pt_waveform_overlays.png.

![Figure 4. TF vs PT waveform overlays for six SeisBench traces (three TXED + three STEAD).](figures/tf_pt_waveform_overlays.png)

## 5. Discussion

### 5.1 Implications for interoperability

Scientific software often inherits framework choices from the era in which a model was first trained. That history is reasonable in isolation, but it sits uneasily next to community standards that consolidate around a single framework for tooling, tutorials, and fair comparison. A documented weight-transfer path does not erase that tension, but it reduces the penalty for institutions that invested in one stack when the broader ecosystem has moved to another. It also avoids an implicit tax in duplicated training and in duplicated carbon cost when the goal is equivalence to an existing checkpoint rather than a new architecture.

Documented parity increases trust: reviewers and operators can see that the PyTorch artifact matches the TensorFlow reference under stated tolerances, not merely that the code runs. That matters for networks that must justify continuity between published results and operational deployment. We hope the present contribution encourages venues to treat careful transfer documentation as valuable alongside novel network designs, particularly when the underlying model already serves real networks.

Beyond numerical equivalence, the converted PyTorch model has measurable performance advantages relevant to the deployments that motivated this work. Figure 5 summarizes a controlled inference benchmark where both TensorFlow and PyTorch versions of the P and S models process ten different 60-second (6000-sample) windows after a brief warmup, on CPU and on each GPU. Panel A reports per-window inference latency, panel B the corresponding single-stream throughput, and panels C–D report the host RAM increase after model load and the peak GPU memory used during inference. The PyTorch path is consistently faster on both CPU and GPU and uses meaningfully less GPU memory; on CPU it also avoids the load-time overhead and visible-device gymnastics that complicate TensorFlow startup in worker processes. These differences are operationally relevant in orchestration frameworks like RAPID (Skevofilax et al., 2026) and SCMLPick (Muñoz et al., 2025), where many short-lived tasks repeatedly load and run a picker and any reduction in per-task setup or per-window latency translates directly into wall-clock catalog throughput.

**Figure 5.** EQCCT TensorFlow versus PyTorch performance on identical 60-second ZNE windows. (A) Mean per-window inference latency (log-scale, ms; bars are the mean over 10 windows after 3 warmup forwards, error bars are the across-window standard deviation) for TF P, TF S, PT P, PT S on CPU, GPU 0, and GPU 1. (B) Corresponding single-stream throughput in windows / second (log scale). (C) Host RAM increase (MB, RSS delta) attributable to loading the TensorFlow models, then to loading the PyTorch models on top. (D) Peak GPU VRAM used during inference for TF and for PT (CPU profile shows 0 / “n/a”). Source data: results/perf_benchmark.json (run eqcct_sb.validation.tf_pt_perf_benchmark); plot generated by scripts/plot_perf_benchmark.py. File: figures/tf_pt_perf_benchmark.png.

![Figure 5. Inference latency, throughput, host RAM, and peak GPU VRAM for the TensorFlow and PyTorch versions of EQCCT.](figures/tf_pt_perf_benchmark.png)

### 5.2 Practical considerations

Several practical issues arise when reproducing or extending this work. Version control is critical: the specific versions of TensorFlow, Keras, CUDA, and GPU drivers should be documented alongside every released checkpoint pair, because even minor version changes can alter multi-head attention naming conventions or break strict weight loading. Environment consistency matters just as much. If a Jupyter notebook runs in a different conda environment than the validation scripts, TensorFlow may report no available GPUs or load a different library build, which can easily be mistaken for modeling errors.

As deep learning frameworks evolve, not every layer conversion will follow a simple one-to-one mapping. Updates to TensorFlow may change attention bias layouts or multi-head attention naming (as seen in Keras 3), requiring corresponding adjustments to the loader. While EQCCT is representative of transformer-based pickers with convolutional front ends, models with different output heads or normalization schemes will need their own transformation tables and validation tests. The key lesson is to build automated layer-by-layer validation early in the process, so that failures can be traced to specific operations rather than appearing as vague mismatches in end-to-end outputs.

### 5.3 When to convert versus retrain

Conversion is most appropriate when the goal is to preserve an existing, field-tested checkpoint and make it usable in a different ecosystem. It is less attractive when the model must be structurally modified (e.g., new heads, new normalization schemes, or changed input sampling) or when the training pipeline is already being revisited for other reasons. In those cases, retraining in the target framework may be simpler than maintaining conversion code that must track two evolving implementations.

### 5.4 Generalizability

The workflow we describe applies most directly to encoder-style phase pickers with static Keras graphs and weights stored as HDF5 checkpoints. EQCCT serves as a useful test case because it combines several complexities: a dual-branch architecture, different depths between branches, and active use in production systems—all of which stress-test the conversion pipeline. The broader principles—establishing canonical module definitions, applying systematic layout transformations, using name-based weight loading in the reference framework, and validating progressively from individual tensors to full waveforms—transfer readily to other institutional models even when the specific layer configurations differ.

### 5.5 Limitations

This study reports numerical agreement between two specific implementations of the same architecture under fixed preprocessing and windowing rules. It does not, by itself, prove that conversions will be equally straightforward for models with dynamic graphs, TensorFlow-only operators, quantized or heavily fused deployments, or architectures that depend on framework-specific behavior. The GPU results also illustrate that device kernels can introduce larger—but still bounded—differences than CPU reference runs. In contexts where bit-level agreement is required, the correct response is to constrain the reference environment (e.g., CPU) or to report device-specific tolerances explicitly, not to assume that any GPU stack will reproduce the trainer exactly.

## 6. Data and code availability

Seismic waveform data and computing resources for network-scale experiments are described in related TexNet publications (Muñoz et al. 2025; Skevofilax et al. 2026). EQCCT training artifacts and architecture definitions are distributed through the open EQCCT project; the Keras checkpoints we convert are from the same lineage as those used in operational studies (Chen et al., 2024). For broader validation, we use publicly available SeisBench datasets including STEAD and TXED (Woollam et al., 2022; Münchmeyer et al., 2022).

The TensorFlow-to-PyTorch conversion code, validation scripts, and demonstration notebooks are available in the EQCCT_to_Seisbench repository under the eqcct_sb package. A standalone methods bundle (methods_tf_to_pt_contribution) provides the same code with documentation for external users. Reference artifacts include Keras HDF5 checkpoints for both P and S branches, along with optional PyTorch checkpoints generated by the validation utilities. Validation experiments were conducted on dual AMD Ryzen Threadripper PRO 7985WX CPUs with 512 GB RAM and two NVIDIA RTX 6000 Ada Generation GPUs, using the CPU affinity controls documented in the RAPID framework.

## 7. Conclusion

Institutional groups should not have to choose between honoring years of TensorFlow training and participating in a PyTorch-centered ecosystem. We have presented a concrete, reproducible workflow for converting EQCCT's Keras checkpoints to PyTorch in a format compatible with SeisBench, and we have demonstrated how to validate the conversion from individual tensor layouts through full waveform tests.

The most valuable technical lessons are procedural: designate one canonical PyTorch module per branch, treat dense layer and multi-head attention layout rules as essential constraints rather than afterthoughts, load Keras reference weights using semantics that match the saved file structure, and build staged validation so that the first incorrect layer is immediately identifiable. While these steps are not unique to EQCCT, the dual-branch architecture and production deployment context made the cost of errors high enough that an explicit, documented workflow became necessary.

Looking forward, seismology will continue to accumulate trained models across different frameworks and hardware generations. Making model portability a routine part of the development process through documented transformations, pinned software environments, and quantitative parity metrics will help the community reuse existing models without costly retraining. As the field increasingly adopts PyTorch as a common framework, consolidating tools like SeisBench and institutional models into a unified ecosystem becomes more feasible. We provide this workflow to help seismology researchers transfer their existing work to PyTorch at their discretion, enabling better integration of community resources. We hope this contribution encourages similar documentation for other widely used phase pickers, so that interoperability becomes a shared standard rather than something that is rarely done.

## Acknowledgments

The authors would like to thank the Texas Seismological Network and Seismology Research (TexNet) group and the State of Texas that provided financial support for this publication under the University of Texas at Austin award #201503664.

---

## References

Abadi, M., A. Agarwal, P. Barham, E. Brevdo, Z. Chen, C. Citro, G. S. Corrado, A. Davis, J. Dean, M. Devin, S. Ghemawat, I. Goodfellow, A. Harp, G. Irving, M. Isard, Y. Jia, R. Jozefowicz, L. Kaiser, M. Kudlur, J. Levenberg, D. Mane, R. Monga, S. Moore, D. Murray, C. Olah, M. Schuster, J. Shlens, B. Steiner, I. Sutskever, K. Talwar, P. Tucker, V. Vanhoucke, V. Vasudevan, F. Viegas, O. Vinyals, P. Warden, M. Wattenberg, M. Wicke, Y. Yu, and X. Zheng (2015). TensorFlow: Large-scale machine learning on heterogeneous systems. Software available from tensorflow.org.

Ali, M. (2023, Jul). Distributed processing using ray framework in python.

Amdahl, G. M. (1967). Validity of the single processor approach to achieving large scale computing capabilities. In *Proceedings of the April 18-20, 1967, Spring Joint Computer Conference, AFIPS ’67 (Spring)*, New York, NY, USA, pp. 483–485. Association for Computing Machinery.

Bates, D. and D. Watts (2008, 05). *Nonlinear Regression Analysis and Its Applications*, pp. 32–66.

Chen, Y., O. M. Saad, A. Savvaidis, F. Zhang, Y. Chen, D. Huang, H. Li, and F. Aziz Zanjani (2024). Deep learning for p-wave first-motion polarity determination and its application in focal mechanism inversion. *IEEE Transactions on Geoscience and Remote Sensing* **62**, 1–11.

Chen, Y., Savvaidis, A., Siervo, D., Huang, D., & Saad, O. M. (2024). Near real-time earthquake monitoring in Texas using the highly precise deep learning phase picker. *Earth and Space Science*, **11**, e2024EA003890. [https://doi.org/10.1029/2024EA003890](https://doi.org/10.1029/2024EA003890)

Choi, S. B. (2025). Design and experimental research of on-device style transfer models based on PyTorch to ONNX. Scientific Reports, 15, Article number: 9345.

Czyzewski, M. A. (2021). Transfer Learning Between Different Architectures Via Weights Injection. arXiv preprint arXiv:2101.02757.

Feng, T., S. Mohanna, and L. Meng (2022). Edgephase: A deep learning model for multi-station seismic phase picking. *Geochemistry, Geophysics, Geosystems* **23**(11), e2022GC010453.

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics* **29**(5), 1189–1232.

Gustafson, J. L. (1988, May). Reevaluating amdahl’s law. *Commun. ACM* **31**(5), 532–533.

Herlihy, M. and N. Shavit (2012). *The Art of Multiprocessor Programming, Revised Reprint*. Morgan Kaufmann.

Johnston, J. and J. DiNardo (1997). *Econometric Methods* (4th ed.). McGraw-Hill.

Lim, C. S. Y., S. Lapins, M. Segou, and M. J. Werner (2025). Deep learning phase pickers: how well can existing models detect hydraulic-fracturing induced microseismicity from a borehole array? *Geophysical Journal International* **240**(1), 535–549, [https://doi.org/10.1093/gji/ggae386](https://doi.org/10.1093/gji/ggae386)

McCool, M., J. Reinders, and A. Robison (2012). *Structured Parallel Programming: Patterns for Efficient Computation*. ITPro collection. Elsevier Science.

Moritz, P., R. Nishihara, S. Wang, A. Tumanov, R. Liaw, X. Liang, M. Elibol, Z. Yang, W. Paul, M. I. Jordan, and I. Stoica (2018). Ray: A Distributed Framework for Emerging AI Applications. In *13th USENIX Symposium on Operating Systems Design and Implementation (OSDI 18)*, Carlsbad, CA, pp. 561–577. USENIX Association.

Mousavi, S., W. Ellsworth, Z. Weiqiang, L. Chuang, and G. Beroza (2020, 08). Earthquake transformer—an attentive deep-learning model for simultaneous earthquake detection and phase picking. *Nature Communications* **11**, 3952.

Mousavi, S. M. and G. C. Beroza (2022). Deep-learning seismology. *Science* **377**(6607), eabm4470.

Muñoz, C., V. Salles, C. Skevofilax, and A. Savvaidis (2025). SCMLPick: A SeisComP Module Implementing Machine Learning Phase Picking for Real-Time Seismic Monitoring. [Manuscript submitted for publication]. Texas Seismological Network, University of Texas at Austin.

Münchmeyer, J., J. Woollam, A. Rietbrock, F. Tilmann, D. Lange, T. Bornstein, T. Diehl, C. Giunchi, F. Haslinger, D. Jozinović, A. Michelini, J. Saul, and H. Soto (2022, 01). Which picker fits my data? a quantitative evaluation of deep learning based seismic pickers. *Journal of Geophysical Research: Solid Earth* **127**.

ONNX Community (2021). ONNX: Open Neural Network Exchange. Available at https://onnx.ai/

Paszke, A., S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, A. Desmaison, A. Köpf, E. Yang, Z. DeVito, M. Raison, A. Tejani, S. Chilamkurthy, B. Steiner, L. Fang, J. Bai, and S. Chintala (2019). PyTorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems 32, 8024–8035.

Ray.io (n.d.). What is ray core? [https://docs.ray.io/en/latest/ray-core/walkthrough.html](https://docs.ray.io/en/latest/ray-core/walkthrough.html). Accessed: 2023-11-13.

Saad, O. M. and Y. Chen (2022). Capsphase: Capsule neural network for seismic phase classification and picking. *IEEE Transactions on Geoscience and Remote Sensing* **60**, 1–11.

Saad, O. M., Y. Chen, A. Savvaidis, W. Chen, F. Zhang, and Y. Chen (2022). Unsupervised deep learning for single-channel earthquake data denoising and its applications in event detection and fully automatic location. *IEEE Transactions on Geoscience and Remote Sensing* **60**, 1–10.

Saad, O. M., Y. Chen, A. Savvaidis, S. Fomel, and Y. Chen (2022). Real-time earthquake detection and magnitude estimation using vision transformer. *Journal of Geophysical Research: Solid Earth* **127**(5), e2021JB023657.

Saad, O. M., Y. Chen, D. Siervo, F. Zhang, A. Savvaidis, G.-c. D. Huang, N. Igonin, S. Fomel, and Y. Chen (2023). Eqcct: A production-ready earthquake detection and phase-picking method using the compact convolutional transformer. *IEEE Transactions on Geoscience and Remote Sensing* **61**, 1–15.

Saad, O. M., G. Huang, Y. Chen, A. Savvaidis, S. Fomel, N. Pham, and Y. Chen (2021). Scalodeep: A highly generalized deep learning framework for real-time earthquake detection. *Journal of Geophysical Research: Solid Earth* **126**(4), e2020JB021473.

Savvaidis, A., B. Young, G. D. Huang, and A. Lomax (2019, 06). Texnet: A statewide seismological network in texas. *Seismological Research Letters* **90**(4), 1702–1715.

SecondTalent (n.d.). Pytorch vs. tensorflow: usage, popularity, and performance. https://www.secondtalent.com/resources/pytorch-vs-tensorflow-usage-popularity-and-performance/. Accessed: 2024-05-22.

Sheen, D.-H., Y.-J. Seong, S. Jung, and J.-h. Park (2023). Real-time application of deep learning seismic phase picker to earthquake monitoring. Poster S31G-0424, *AGU Fall Meeting*, San Francisco, CA, 11–15 December, session *Machine Learning–Driven Analysis of Geophysical Signals III*.

Skevofilax, C., and A. Savvaidis (2026). RAPID: A Generalized High-Performance Parallelization Framework for Real-Time Deep Learning Seismic Phase Picking. [Unpublished manuscript]. Texas Seismological Network, University of Texas at Austin.

Tan, Y. J., F. Waldhauser, W. Ellsworth, M. Zhang, Z. Weiqiang, M. Michele, L. Chiaraluce, G. Beroza, and M. Segou (2021, 04). Machine-learning-based high-resolution earthquake catalog reveals how complex fault structures were activated during the 2016–2017 central italy sequence. *The Seismic Record* **1**, 11–19.

Verma, G., S. Raskar, M. Emani, and B. Chapman (2024). Cross-Feature Transfer Learning for Efficient Tensor Program Generation. Applied Sciences, 14(2), 513.

Wang, H., J. Gu, and Y. Li (2022). An Empirical Study of Challenges in Converting Deep Learning Models. In Proceedings of the 31st ACM SIGSOFT International Symposium on Software Testing and Analysis (ISSTA '22), 206–217.

Wang, T., D. Trugman, and Y. Lin (2021). Seismogen: Seismic waveform synthesis using gan with application to seismic data augmentation. *Journal of Geophysical Research: Solid Earth* **126**(4), e2020JB020077.

Woollam, J., J. Münchmeyer, F. Tilmann, A. Rietbrock, D. Lange, T. Bornstein, T. Diehl, C. Giunchi, F. Haslinger, and D. Jozinović (2022). SeisBench—A Toolbox for Machine Learning in Seismology. *Seismological Research Letters* **93**(3), 1695–1709.

Xiao, Z., J. Wang, C. Liu, J. Li, L. Zhao, and Z. Yao (2021). Siamese earthquake transformer: A pair-input deep-learning model for earthquake detection and phase picking on a seismic arrival-time picking method. *Journal of Geophysical Research: Solid Earth* **126**(5), e2020JB021444.

Yeck, W. L., J. M. Patton, Z. E. Ross, G. P. Hayes, M. R. Guy, N. B. Ambruz, D. R. Shelly, H. M. Benz, and P. S. Earle (2020). Leveraging Deep Learning in Global 24/7 Real-Time Earthquake Monitoring at the National Earthquake Information Center, *Seismol. Res. Lett.* **92**, 469–480, doi: 10.1785/0220200178.

Yu, Z., Y. Jiang, X. Jing, and H. Zheng (2024). Study on geomagnetic observations associated with three major earthquakes in southwest china through a novel deep learning framework. *IEEE Transactions on Geoscience and Remote Sensing* **62**, 1–18.

Zhu, W. and G. C. Beroza (2018, 10). Phasenet: a deep-neural-network-based seismic arrival-time picking method. *Geophysical Journal International* **216**(1), 261–273.
