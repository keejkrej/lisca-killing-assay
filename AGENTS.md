# lisca-killing-assay

## Fleet

PhD work is a multi-repo, multi-machine fleet. Before choosing a machine, cloning, or
moving files, read `~/workspace/phd-notes/standard/README.md`. Status:
`~/workspace/phd-notes/projects/lisca-killing-assay.md`. Prefer `nv5090` for training
and Fig. 6 regeneration. Train-at-scale: `lsr-ex-dgx1`.

## Purpose

Python CLI `apoptosis` and Rust crate `lisca-killing`.

Label-free path: label death frames → train ResNet viability → compare
morphology death time with TOTO-3. Supplies LISCA review Fig. 6 D–E via
`scripts/plot_fig6.py`. EmbeddingGemma and the viable/dead SVM register with the LiSCA inference host.
The host process stays in `keejkrej/lisca` (`docs/adr/0003-assay-registers-its-model.md`).
The offline embedding comparison stays here. Dual-marker LNP event times
(Fig. 6 A–C) are not this repo.

Studio death-reporter fluorescence and fluorescent engagement counts live in
this repo (`lisca-killing`, plus `apoptosis fluorescence` / `apoptosis engagement`).
`keejkrej/lisca` imports the crate at build time and keeps scheduling, progress,
and the shared figures. Studio UI stays in `keejkrej/lisca`.

## Commands

```sh
uv sync
uv run apoptosis --help
```

Typical pipeline: `label` → `dataset-build` → `train` → `eval` → `predict`.

## Out of scope

Studio UI, transfection/binding analysis, review-paper prose.
