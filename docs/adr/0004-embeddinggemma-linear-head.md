# ADR-0004: Label-free viability uses EmbeddingGemma and a linear head

- **Status:** accepted
- **Date:** 2026-10-20

## Decision

Label-free viability in this package is a frozen EmbeddingGemma 2 encoder plus a
linear SVM head trained on a small labelled set. That is the only local
train/eval/predict path.

## Why

A small head on frozen embeddings can be fit from the labels a person actually
has time to make. Encoder weights stay pinned in
`killing.inference.embeddinggemma`. The CLI writes `runs/viability/head.json`.
The LiSCA inference host still loads this encoder through ADR-0003.

Studio death-reporter fluorescence, fluorescent engagement, and
`lisca_killing::run_predict_to` (`model.onnx`, `onnx` feature) are unchanged.

## Looks like a bug when

- Encoder weights are trained or a second local morphology classifier is added.
- `killing train` writes anything other than the linear head `head.json`.
- Predict loads a different encoder revision than the saved head.
