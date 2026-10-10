# Compare ROI classifications using frozen embeddings

The offline experiment in `scripts/compare_roi_embeddings.py` compares
cosine nearest neighbors with a class-balanced linear SVM on the same frozen image
embeddings. It does not fine-tune the encoder or change Studio's analysis pipeline.
Run embedding on the GPU host; the classifier comparison only needs a CPU.

## Figure 6 killing: existing labels, 2026-10-06

This is the primary comparison. The existing
`lisca-killing-assay/datasets/viability/manifest.json` points to
`/home/jack/data/lisca_review/fig6/20260327` and contains manual death-frame-derived
labels for 50 ROI sequences. Its original split is retained: 40 reference ROIs and
10 validation ROIs. Frames from a given ROI never cross that boundary. This uses
the existing annotations, not ResNet predictions or Codex visual labels.

Every tenth stored frame was embedded: 1,250 brightfield images, including 1,000
references (717 viable, 283 dead) and 250 validation images (201 viable, 49 dead).
The data includes Positions 0, 28, and 70. Preprocessing uses the original killing
per-frame 1st/99th percentile normalization, with uint8 quantization for the image
encoder. The reporter channel is not an input. EmbeddingGemma 2 stays frozen at
revision `914f7f89142e33e77833254d9c9b90c3cef7303b`, producing 768-dimensional vectors
on nv5090's RTX 5090. No foundation-model training or fine-tuning was performed.

| Method                   | Accuracy | Balanced accuracy | Labeled-dead detected | Viable called dead |
| ------------------------ | -------- | ----------------- | --------------------- | ------------------ |
| Always predict viable    | 80.4%    | 50.0%             | 0/49                  | 0/201              |
| Cosine kNN, k=3          | 82.4%    | 75.9%             | 32/49                 | 27/201             |
| Balanced linear SVM, C=1 | 80.8%    | 85.0%             | 45/49                 | 44/201             |

Balanced accuracy here is the mean recall across the two classes. The SVM detects
more labeled-dead frames but makes more false death calls. The 80.4% majority-class
baseline illustrates why overall accuracy alone is a poor choice for this split.
These are frame-label agreement metrics on ten validation ROIs from one experiment,
not independent reporter validation, cross-experiment performance, or death-time
accuracy. Neighbor vote support and SVM margins are not calibrated probabilities.

Class-balanced reference subsets were also compared with the same validation ROIs
and fixed classifier settings. Each row below averages five random seeds (42–46);
subsets within each seed are nested. Labels were used to balance the reference
selection, and multiple reference images may belong to one ROI. This is an
exploratory annotation-budget comparison, not five independent validation sets.

| Reference images | kNN balanced accuracy | SVM balanced accuracy |
| ---------------- | --------------------- | --------------------- |
| 8                | 75.0%                 | 76.1%                 |
| 16               | 74.7%                 | 75.7%                 |
| 32               | 75.6%                 | 74.5%                 |
| 64               | 76.9%                 | 78.2%                 |
| 128              | 78.2%                 | 80.7%                 |
| 256              | 78.6%                 | 81.8%                 |

The full-reference row uses the natural class distribution; the smaller subsets
are balanced. Increasing labels does not guarantee monotonic improvement. This
establishes a working frozen-encoder baseline, with a clear sensitivity/specificity
trade-off; it does not demonstrate that a few examples suffice for deployment.

Artifacts are at `nv5090:~/data/fig6-apoptosis-embedding-20261006/`, with the manifest,
vectors, metadata, comparison, and reference-size report copied to
`C:/Users/ctyja/data/fig6-apoptosis-embedding-20261006/`. The source TIFFs and original
labels remain unchanged. `comparison.json` preserves the labels used for each run.

To reproduce on the data host after installing dependencies as below:

```sh
uv run --no-sync python scripts/compare_roi_embeddings.py import-frames \
  ~/workspace/lisca-killing-assay/datasets/viability/manifest.json \
  ~/data/fig6-apoptosis-embedding-new --frame-stride 10 \
  --label-map '{"0":"dead","1":"viable"}' \
  --label-source existing-apoptosis-manual-death-frame-manifest
uv run --no-sync python scripts/compare_roi_embeddings.py embed \
  ~/data/fig6-apoptosis-embedding-new --batch-size 16
uv run --no-sync python scripts/compare_roi_embeddings.py compare \
  ~/data/fig6-apoptosis-embedding-new
uv run --no-sync python scripts/benchmark_reference_sizes.py \
  ~/data/fig6-apoptosis-embedding-new
```

The initial run was prepared with a one-off adapter equivalent to `import-frames`;
the source manifest hash is recorded. The importer expects `data_dir` and `samples`
with `position`, `roi_id`, `time_index`, `label`, and `split` (`train` or `val`). It
preserves label values through an explicit mapping and rejects ROI split leakage.

## Causal temporal comparison, 2026-10-06

Using the same cached vectors and original ROI split, three fixed feature choices
were compared with k=3 and C=1. The encoder remains frozen. A trailing mean averages
the current and previous two sampled embeddings, then normalizes the result. At
stride 10 this spans 20 acquisition frames; early observations use the available
history. Baseline delta concatenates the current vector with its difference from
the earliest available vector of that ROI, then normalizes. The baseline is not
assumed to represent a viable cell. Neither method reads future frames or labels.

| Features | Classifier | Accuracy | Balanced accuracy | Labeled-dead detected | Viable called dead |
| -------- | ---------- | -------- | ----------------- | --------------------- | ------------------ |
| Single frame | kNN | 82.4% | 75.9% | 32/49 | 27/201 |
| Trailing mean, 3 observations | kNN | 89.2% | 84.8% | 38/49 | 16/201 |
| Baseline delta | kNN | 88.0% | 74.0% | 25/49 | 6/201 |
| Single frame | SVM | 80.8% | 85.0% | 45/49 | 44/201 |
| Trailing mean, 3 observations | SVM | 78.8% | 81.4% | 42/49 | 46/201 |
| Baseline delta | SVM | 85.2% | 83.1% | 39/49 | 27/201 |

The trailing mean improves kNN's sensitivity and specificity in this comparison,
without new neural inference on the cached images. It does not improve the SVM.
All 16 remaining false death calls for trailing-mean kNN occur in Pos0/Roi17
(six frames) and Pos0/Roi26 (ten frames). All four manually labeled surviving
validation ROIs have no errors with this method. The 11 missed dead frames are
distributed across five ROIs. The timelines show errors away from the annotated
death transition too; this is not merely uncertainty in the exact event frame.
Visual review includes crops with multiple cell bodies, but does not establish
that the original labels are wrong. No labels were changed.

This is exploratory reuse of the same ten validation ROIs, not an independent
test of the preferred feature choice. Correlated frames do not provide 250
independent biological observations. Causal averaging can delay a transition;
improved frame classification does not establish improved death-time estimation.
No event detector or death-time metric was fitted here. Whole-experiment testing
and review of the two false-positive ROIs remain necessary before deployment.

Reproduce on the data host with the optional experiment dependencies installed
(including matplotlib for plots):

```sh
uv run --no-sync python scripts/analyze_embedding_context.py \
  ~/data/fig6-apoptosis-embedding-20261006
```

Outputs under `context-comparison/` include per-method predictions, confusion
matrices, per-ROI errors, a summary, representative single-frame error contact
sheets, and `validation-trajectories.png` for all ten held-out ROIs. These are
copied to the local artifact directory listed above.

### Single-ROI step visualization

`scripts/plot_roi_viability.py DIRECTORY --group Pos0/Roi12` plots cached
single-frame and trailing-mean kNN viable vote support, sampled manual labels,
and a retrospective viable-to-dead step. The step minimizes squared error against
trailing-mean vote support over every possible change index, including constant
viable and constant dead states. It does not use manual labels for fitting. This
is a whole-sequence fit, not an online detector, and imposes at most one transition.

Pos0/Roi12 (the first validation ROI in the manifest, not selected for best fit)
switches at sampled frame 200. Its manual death-frame annotation is 157 (first
sampled dead label at 160). The inferred transition is therefore late; a clean
step must not be mistaken for accurate event timing. Squared error is 0.938 versus
4.579 for the best constant state, without a significance interpretation.
The output includes `viability.png`, `trajectory.json`, and `sampled-movie.gif`
under `single-roi/Pos0-Roi12/`. The animation contains every tenth frame and uses
arbitrary playback timing. Displayed crops use the embedding input's per-frame
contrast normalization.

## Inference speed, 2026-10-06

`scripts/benchmark_embedding_speed.py DIRECTORY` measured the same frozen model
revision in float32 on nv5090's RTX 5090 (PyTorch 2.12.1+cu130). Thirty-two existing
RGB images were selected with seed 42. Each batch size was warmed independently;
three timed passes used CUDA synchronization, reporting median throughput.

| Batch size | Images/second | Amortized ms/image | Peak allocated GPU GiB |
| ---------- | ------------- | ------------------ | ---------------------- |
| 1 | 15.5 | 64.6 | 2.95 |
| 8 | 24.3 | 41.2 | 3.92 |
| 16 | 23.1 | 43.2 | 5.02 |
| 32 | 23.7 | 42.2 | 6.98 |

These timings include image processing inside `encode`, CPU/GPU transfers, model
inference, normalization, and returning NumPy vectors. They exclude TIFF reading,
ROI cropping/contrast conversion, network transfer, queueing, and cache writes.
Amortized per-image cost is not an individual request's latency in a batch.
Loading cached model weights took 4.69 seconds; opening/converting the 32 PNGs
took 20.8 ms in this run. Neither is a cold-disk benchmark.

For 250 cached queries against 1,000 reference vectors, CPU kNN cost 0.064 ms per
query, using cosine similarities, stable sorting and inverse-distance votes. SVM
prediction cost 0.00058 ms per query; fitting the small SVM took 34.9 ms. CPU timings
are medians over 20 passes with one BLAS thread and exclude image embedding and
report generation. Embedding dominates these measured stages.

Batching eight images gave about 1.57 times the single-image throughput; larger
batches did not improve this short benchmark. At 24.3 images/second, 1,000 ROI
images would take roughly 41 seconds of warm embedding processing, and a 241-frame
movie roughly 10 seconds, extrapolating to similar inputs. Separate GPU workers
can partition independent images, but no multi-GPU or concurrent-worker scaling
was measured. DGX Spark, reduced precision, and end-to-end server latency remain
unmeasured. Results are saved as `speed-benchmark.json` in the artifact directory.

## Run

From the repository root on the host with the data:

```sh
uv sync
uv pip install --python .venv -r requirements-embedding.txt
uv run --no-sync python scripts/compare_roi_embeddings.py sample \
  ~/data/experiment ~/data/embedding-comparison \
  --roi-count 24 --frames-per-roi 3 --channel 0 --seed 42
uv run --no-sync python scripts/compare_roi_embeddings.py embed ~/data/embedding-comparison
uv run --no-sync python scripts/compare_roi_embeddings.py sequences ~/data/embedding-comparison
```

Sampling requires LiSCA's `roi/Pos*/index.json` metadata and `TCZYX` stacks. It
reads individual TIFF pages, never all the acquisition into memory. The selected
channel and Z plane are explicit. It samples ROIs uniformly, then frames uniformly
within each ROI; this is not uniform sampling over every image in the acquisition.
Seventy percent of the sampled ROIs become references, with the rest reserved for
evaluation before labeling. All frames from a given ROI stay in one split.

The output directory contains contact sheets, individual PNGs, and `manifest.json`.
The `sequences` command also produces time-ordered strips at fixed contrast, with
the selected images marked. Review these before labeling: crops can lose a cell
through drift rather than death. Original acquisition frame IDs are preserved
from `timeIndices`, separately from the stored stack frame index.
Review the images and set each row's `label`, `label_source`, and `notes`. Use
`uncertain` or null for images you cannot label; they do not become references or
contribute to metrics. Other nonempty labels are class names. Labels never go into
the image encoder. Record whether labels are expert annotations, reporter-derived,
or provisional visual judgments.

```sh
uv run --no-sync python scripts/compare_roi_embeddings.py compare ~/data/embedding-comparison \
  --neighbors 3 --svm-c 1
```

The comparison writes `comparison.json`: predictions for every evaluation image,
nearest reference IDs and cosine similarities, kNN vote support, SVM decision
scores, confusion matrices, and the IDs actually scored. Unlabeled evaluation
images get predictions but no invented accuracy. Vote support and SVM decision
scores are not calibrated probabilities, and neither classifier automatically
abstains. The `uncertain` label is a review decision, not a learned class.

The encoder defaults to `google/embeddinggemma-2`, float32, with a batch size of
eight. By default, each ROI uses the first stored frame's first and 99th intensity
percentiles as fixed limits for all sampled frames. Images are clipped to those
limits, converted to uint8, and replicated across RGB channels. This preserves
darkening relative to that ROI's initial appearance; it does not preserve absolute
intensity across ROIs. `sample --contrast frame` instead scales each frame
independently and can obscure temporal darkening. Inspect the previews before
choosing preprocessing for another task.
The model's own processor handles resizing. Image inputs use the documented
`{"image": image}` interface without text labels or task prompts.

Embeddings are cached independently of labels. Changing labels only reruns the
small classifiers. Metadata records image hashes, model revision, preprocessing,
and image order; changed image bytes invalidate the cache. Use the same encoder
revision for references and queries. Weights stay in the Hugging Face cache, and
all experiment artifacts stay outside git under `~/data/`.

## Requested Zeiss dataset: invalid crops, 2026-10-06

The user supplied `C:/Users/ctyja/data/jb4_zeiss_portable_exceptraw` on `nv3090`.
Seventy-two images from 24 ROIs were sampled with seed 42. Only the sampled PNGs
were copied to `nv5090` for GPU embedding; the original TIFFs were not modified.
Artifacts are in `C:/Users/ctyja/data/jb4-zeiss-ring-loss-20261006`, with a matching
GPU-side directory under `/home/jack/data/`.

The user clarified the phenotype for this cell line: killed tumor cells mostly
darken and lose the original phase-contrast ring while remaining adherent to the
Pattern. Disappearance is not the required readout. Fixed-contrast sequence review
then showed a common apparent ring-loss interval around source Frame 350–375,
including Pos91 / Sample `0_1`, which the user confirmed is the no-T-cell control.

The bright cell moves out through the top of multiple control crops. In
Pos91/Roi0, a fixed-threshold bright-component diagnostic measured its vertical
centroid at approximately 56 pixels at Frame 0, 27 at Frame 250, 13 at Frame 335,
and 2 at Frame 350 in a 105-pixel-high crop. It touches the top boundary from
Frame 335. Similar motion occurs in other control ROIs. The align files have no
saved drift correction. The user confirmed that this dataset's crops are broken
by drift and will provide a Nikon dataset later.

Both classifiers reproduced the provisional visible-ring labels on all 18
scored evaluation images (15 ring-not-visible, three ring-present; six uncertain
excluded). **This is artifact classification, not evidence of killing detection.**
The same eight evaluation ROIs were reused after revising the labeling criterion,
so even these agreement counts are exploratory validation, not an untouched test.
The earlier per-frame-contrast run and its absence-based labels are superseded.
Do not use any of these Zeiss results to select a biological classifier or infer
death times. The artifact manifests and comparison report mark the evaluation
as invalid for biological conclusions. No drift correction was attempted because
the source acquisition is absent from this portable folder.

The user then requested the already-labeled Figure 6 killing data instead; that
comparison is reported above. The Nikon dataset remains a future evaluation. Check
cell retention and controls before interpreting its predictions biologically.

## Preliminary review-data experiment, 2026-10-06

- Host: `nv5090`, RTX 5090. Spark SSH was unavailable during the run.
- Input: `/home/jack/data/lisca_review/fig6/20260327`, channel 0, Z 0.
- Output: `/home/jack/data/lisca-embedding-experiment-20261006`.
- Encoder: `google/embeddinggemma-2`, revision
  `914f7f89142e33e77833254d9c9b90c3cef7303b`, 768 dimensions, frozen.
- Seed 42: 72 frames from 24 ROIs across Positions 0, 28, and 70.
- Preprocessing: independently scaled frames (the original exploratory setting).
- References: 16 ROIs; 35 labeled images and 13 uncertain images.
- Evaluation: eight disjoint ROIs; 18 labeled images and six uncertain images.
- Labels: Codex visual judgments recorded before classifier predictions, using
  `healthy_appearance`, `death_like`, and `uncertain`. Existing assay labels and
  the reporter channel were not used.

| Classifier                              | Agreement with provisional labels | Death-like matched | Healthy appearance matched |
| --------------------------------------- | --------------------------------- | ------------------ | -------------------------- |
| Cosine kNN, k=3, inverse-distance votes | 17/18                             | 9/9                | 8/9                        |
| Balanced linear SVM, C=1                | 18/18                             | 9/9                | 9/9                        |

These are exploratory agreement counts, not biological death-detection accuracy.
Frames within each evaluation ROI remain correlated. This small sample comes from
one experiment, excludes ambiguous images from scoring, and uses one AI reviewer's
unverified morphology labels. It does not establish that the SVM is superior or
that either approach generalizes to another microscope or experiment. Death-like
morphology can be confused with division, interactions, or imaging effects.

Before making biological accuracy claims, obtain independent expert or reporter
labels and evaluate whole held-out experiments. Repeatedly correcting references
against this evaluation set turns it into validation data; a final test needs new,
untouched ROI groups. This tool does not infer death-event timing from sequences.

## Sources

- [EmbeddingGemma 2 overview](https://ai.google.dev/gemma/docs/embeddinggemma)
- [Google's multimodal inference guide](https://ai.google.dev/gemma/docs/embeddinggemma/multimodal-embeddinggemma-with-sentence-transformers)
- [ADR-0006](../adr/0006-frozen-reference-classification.md)

## Validation of the implementation

- Focused Python regression tests pass (18 tests), including frozen inference,
  label-independent cache reuse, changed-image invalidation, fixed contrast,
  acquisition-frame mapping, rejection of ROI leakage in imported splits, causal
  temporal context, row-order preservation, and channel/group isolation.
- Ruff and targeted `ty` checks pass.
- The broader Windows Python suite has an existing failure in
  `test_crop_fd_budget_reads_rlimit` because the Unix `resource` module is absent.
- `pnpm run check` reaches UI tests and fails on Windows resolving
  `file:///@solid-refresh`; this change does not modify the UI or its tooling.
