# ADR-0002: Serve frozen viability inference from this package

- **Status:** superseded
- **Superseded by:** ADR-0003
- **Date:** 2026-10-06

## Decision

Run the frozen encoder and viable/dead reference classification in this package
as one GPU process, because the encoder dominates cost and Studio should only
dial the service.

## Why

A resident worker, shared batching, duplicate suppression, and a persistent
pixel/revision cache avoid repeated model startup and redundant work. One process
per GPU preserves batching; more HTTP workers on the same GPU would duplicate
weights. Float32 and a pinned model revision preserve the experiment's basis
while future throughput changes are measured.

Studio in `keejkrej/lisca` opens this connection explicitly (its ADR-0007). The
workspace API stays in that repo. Tailscale Serve is private; Funnel is public
and is not enabled. The token is checked on every API route.

Single-frame predictions are the default because temporal averaging delayed the
clean single-cell example. A separately labeled retrospective step can summarize
the movie, but does not establish accurate death timing. Ambiguous TIFF page axes
require the ROI index; guessing would mix reporter channels with time.

The ResNet `predict` command is not redirected here: its checkpoint contract and
outputs differ.

## Looks like a bug when

- A movie with no death is forced into a transition instead of keeping a constant viability state.
- Extra HTTP workers are added on one GPU to raise throughput.
- A plain ROI TIFF is accepted by guessing which pages are time and which are the reporter channel.
