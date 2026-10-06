# ADR-0003: Register EmbeddingGemma and the viability classifier with the host

- **Status:** accepted
- **Date:** 2026-10-06

## Decision

This package registers EmbeddingGemma and the viable/dead classifier with the
LiSCA inference host, because the model and the SVM are assay-specific and the
batching service is not.

## Why

ADR-0002 ran the HTTP process here. The host in `keejkrej/lisca` now owns
batching, the cache, and authorization so another model can use the same
process. This package still chooses the encoder revision and the viable/dead
task, including the offline SVM comparison. A model trained for this assay
would replace EmbeddingGemma here, not in the host.

The ResNet `predict` command is not redirected to this encoder.

## Looks like a bug when

- The HTTP server, queue, or embedding cache is implemented in this package.
- Viable/dead fitting is imported from the LiSCA host.
- A movie with no death is forced into a transition instead of keeping a constant viability state.
