# lisca-killing-assay

## Fleet

PhD work is a multi-repo, multi-machine fleet. Before choosing a machine, cloning, or
moving files, read `~/workspace/phd-notes/standard/README.md`. Status:
`~/workspace/phd-notes/projects/lisca-killing-assay.md`. Prefer `nv5090` for training
and Fig. 6 regeneration. Train-at-scale: `lsr-ex-dgx1`.

## Purpose

Rust crate `lisca-killing`, imported by Lisca. The Python package `killing` is the definition library and the local training CLI.

Label-free path: label death frames → train ResNet viability → compare
morphology death time with TOTO-3. Supplies LISCA review Fig. 6 D–E via
`scripts/plot_fig6.py`. EmbeddingGemma and the viable/dead SVM register with the LiSCA inference host.
The host process stays in `keejkrej/lisca` (`docs/adr/0003-assay-registers-its-model.md`).
The offline embedding comparison stays here. Dual-marker LNP event times
(Fig. 6 A–C) are not this repo.

Studio death-reporter fluorescence, fluorescent engagement, and the killing
classifier (predict, monotonicity clean, kill curve) live here as library code
(`lisca-killing`, `killing.core`, `killing.services`). `keejkrej/lisca` imports
the crate at build time. Studio UI, scheduling, figures, and `lisca-analyze`
stay in `keejkrej/lisca`. `lisca_killing::run_predict_to` loads `model.onnx`
behind the crate's `onnx` feature. Python `predict` is the Lightning checkpoint
tool, not that function.

## Commands

```sh
uv sync
uv run killing --help
```

Typical pipeline: `label` → `dataset-build` → `train` → `eval` → `predict`.

## Out of scope

Studio UI, transfection/binding analysis, review-paper prose.
