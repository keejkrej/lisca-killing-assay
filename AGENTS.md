# lisca-killing-assay

## Fleet

PhD work is a multi-repo, multi-machine fleet. Before choosing a machine, cloning, or
moving files, read `~/workspace/phd-notes/standard/README.md`. Status:
`~/workspace/phd-notes/projects/lisca-killing-assay.md`. Prefer `nv5090` for training
and Fig. 6 regeneration. Train-at-scale: `lsr-ex-dgx1`.

## Purpose

Python CLI `apoptosis`: label death frames → train ResNet viability → compare
morphology death time with TOTO-3. Supplies LISCA review Fig. 6 D–E via
`scripts/plot_fig6.py`. The frozen-encoder viability service and the offline
embedding comparison also live here (`docs/inference-server.md`,
`docs/adr/0001-frozen-reference-classification.md`). Dual-marker LNP event times
(Fig. 6 A–C) are not this repo. Studio UI stays in `keejkrej/lisca`.

## Commands

```sh
uv sync
uv run apoptosis --help
```

Typical pipeline: `label` → `dataset-build` → `train` → `eval` → `predict`.

## Out of scope

Studio UI, transfection/binding analysis, review-paper prose.
