# killing

Library Lisca imports (`lisca-killing`) and the local label-free training tool. It is not the program that runs a Studio assay. The two killing kinds below are training-package kinds on LiSCA ROI time-lapses (one ROI per micropattern Pattern).

| Kind | Id | Signal |
| --- | --- | --- |
| Death reporter | `death-reporter` | A fluorescent reporter of a death event. The reporter is the signal, as in a transfection Trace. |
| Label-free | `label-free` | Brightfield or phase contrast. Manual labels train a ResNet; there is no death reporter. |

The commands below are the label-free path. They compare morphology-based death timing with TOTO-3, which is one death reporter. Ids live in `killing.core.assay`.

Death-reporter fluorescence and fluorescent engagement are the Studio measurements. Rust crate `lisca-killing` is what Lisca imports. Lisca writes every PNG and the death-reporter `traces.xlsx`. The crate writes fluorescence CSVs (`analysis/Pos{n}/ch{m}.csv`: `roi,t,area,background,sum,corrected`), `engagement.csv`, `engagement_summary.csv`, and the engagement workbooks.

`lisca-analyze killing` and `lisca-analyze killing-engagement` in [keejkrej/lisca](https://github.com/keejkrej/lisca) run those measurements. They arrive with Lisca PR 159 and are not on `lisca` `main` until that PR merges.

A crop stays alive while `p_dead` is below 0.5; once it is dead, later alive labels are cleared. Python `predict` loads a Lightning checkpoint and writes `runs/viability/inference.json`. That is not `lisca_killing::run_predict_to`, which loads `model.onnx` behind the crate's `onnx` feature.

## Install

```bash
uv sync
```

The `killing` entry point is available in the project virtual environment.

```bash
uv run killing --help
uv run killing <command> --help
```

## Input layout

Commands expect a LiSCA Workspace (`--data-dir`) with per-Position ROI stacks from ROI crop. Each ROI TIFF interleaves brightfield (channel 0) and Toto-3 (channel 1) pages, two pages per Frame:

```text
data_dir/
  roi/
    Pos0/
      Roi0.tif
      Roi1.tif
      ...
    Pos28/
      Roi0.tif
      ...
```

Manual labels are stored as JSON (default: `<repo>/labels.json`):

```json
[
  {
    "position": "Pos0",
    "roi_id": 0,
    "death_frame": 106,
    "labeled_at": "2026-06-28T20:10:53.364121+00:00"
  }
]
```

`death_frame` is a Frame index. A `death_frame` equal to the ROI Frame count means the cell stayed healthy through the acquisition.

## Workflow

Typical end-to-end pipeline:

```text
label -> dataset-build -> train -> eval -> predict
```

```bash
# Annotate ROIs in the browser (writes labels.json)
uv run killing label --data-dir /path/to/data

# Build per-frame train/val manifest from labels
uv run killing dataset-build --data-dir /path/to/data --labels-path labels.json

# Train ResNet viability classifier
uv run killing train --manifest datasets/viability/manifest.json

# Report accuracy/F1 on train and val splits
uv run killing eval --checkpoint runs/viability/lightning_logs/.../best-*.ckpt

# Infer all cells and write validation figure
uv run killing predict --data-dir /path/to/data
```

**Outputs**

| Step | Output |
|---|---|
| `label` | `labels.json` (or `--labels-path`) |
| `dataset-build` | `datasets/viability/manifest.json` |
| `train` | `runs/viability/lightning_logs/version_*/checkpoints/best-*.ckpt` |
| `eval` | Metrics printed to stdout |
| `predict` | `runs/viability/inference.json`, `runs/viability/validation_plot.png` |

## Labeling web app

`killing label` starts a FastAPI server (`killing.api`) backed by `routes/labeling.py`. The browser UI (`static/label.html`) lists ROIs, shows brightfield/Toto-3 frames, and saves death-frame annotations via REST (`/api/rois`, `/api/labels`). Open the URL printed by the command (default `http://127.0.0.1:8000`).

## Command reference

| Command | Role |
|---|---|
| `label` | Launch the ROI viability labeling webapp |
| `dataset-build` | Build a per-frame viability dataset manifest from manual labels |
| `train` | Train a ResNet viability classifier with PyTorch Lightning |
| `eval` | Evaluate a trained model on train and val splits |
| `predict` | Lightning checkpoint inference; write `runs/viability/inference.json`. Not `lisca_killing::run_predict_to` |
| `hello` | Greet someone (smoke test) |
| `version` | Show the installed version |

## Frozen-encoder viability

Label-free viability can also be scored with a frozen EmbeddingGemma encoder,
cosine nearest neighbors, and a small linear SVM. No encoder weights are trained.
The offline comparison is `scripts/compare_roi_embeddings.py`
([results](docs/roi-embedding-comparison.md)). EmbeddingGemma and the
viable/dead classifier register with the LiSCA inference host
(`python -m lisca.inference.server` in
[keejkrej/lisca](https://github.com/keejkrej/lisca)). Studio dials that host.
Optional encoder install: `requirements-inference.txt`. The offline comparison
also uses `requirements-embedding.txt`.

## Paper figures

One-off figure scripts that are not CLI commands live in `scripts/`. For example, `scripts/plot_fig6.py` reads `runs/viability/inference.json` and writes paper figures to an external repo path configured in the script.