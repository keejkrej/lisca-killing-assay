# ADR-0001: Compare label-free viability with a frozen encoder

- **Status:** accepted
- **Date:** 2026-10-06

## Decision

We will evaluate label-free viability through an offline tool using frozen image
embeddings, cosine nearest neighbors, and a linear SVM, because users should adapt
classification by labeling examples without fine-tuning a foundation model.

## Why

Embedding generation is the GPU operation; reference edits reuse those vectors.
The small SVM is fitted on labeled embeddings, while no encoder weights change.
The comparison lives in this assay because the labels are viable and dead on
brightfield ROIs, scored against this package's viability manifest. It does not
add weights to the product `models/` tree and does not replace the ResNet
training commands.

Randomly splitting time-lapse frames leaks the same ROI into both reference and
evaluation sets. Split whole ROIs before labeling, and keep ambiguous annotations
out of metrics. AI visual labels measure agreement with that reviewer, not
independently established cell death.

Default image normalization holds each ROI's first-frame contrast limits fixed,
because scaling frames independently can hide the darkening and phase-ring loss
identified by the user for this cell line. Sequence contact sheets expose drift
and other changes that can masquerade as death. Disappearance and ring loss are
not automatically mapped to a biological death label.

The LiSCA host loads this encoder through ADR-0003. This comparison does not
replace `killing predict`.

## Looks like a bug when

- Adding a label does not update the encoder or regenerate unchanged embeddings.
- SVM fitting happens despite the encoder being described as frozen.
- An uncertain evaluation image has a prediction but is excluded from accuracy.
- Frames from one ROI cannot be moved independently between the two splits.
