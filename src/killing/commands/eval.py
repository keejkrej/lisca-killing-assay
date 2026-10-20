from pathlib import Path

import typer

from killing.app import app
from killing.core.session import PROJECT_ROOT
from killing.services.dataset_build import default_dataset_dir
from killing.services.evaluate import evaluate_head
from killing.services.head_train import default_head_path


@app.command()
def eval(
    head: Path = typer.Option(
        default_head_path(PROJECT_ROOT),
        "--head",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    manifest_path: Path = typer.Option(
        default_dataset_dir(PROJECT_ROOT) / "manifest.json",
        "--manifest",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    embeddings_path: Path | None = typer.Option(
        None,
        "--embeddings",
        dir_okay=False,
        resolve_path=True,
        help="Optional cached embeddings (.npz). Default: <head-dir>/embeddings.npz.",
    ),
    batch_size: int = typer.Option(8, min=1),
    device: str = typer.Option("cuda", help="cuda or cpu for EmbeddingGemma"),
) -> None:
    """Evaluate a trained linear head on train and val splits."""
    cached = embeddings_path
    if cached is None:
        sibling = head.parent / "embeddings.npz"
        cached = sibling if sibling.is_file() else None
    typer.echo(f"Head: {head}")
    results = evaluate_head(
        head_path=head,
        manifest_path=manifest_path,
        embeddings_path=cached,
        batch_size=batch_size,
        device=device,
    )
    for metrics in results:
        typer.echo("")
        typer.echo(f"[{metrics.split}] {metrics.samples} frames")
        typer.echo(
            f"  accuracy={metrics.accuracy:.3f}  "
            f"f1={metrics.f1:.3f}  "
            f"precision={metrics.precision:.3f}  "
            f"recall={metrics.recall:.3f}"
        )
        typer.echo(
            f"  ground truth: {metrics.true_viable} viable / {metrics.true_dead} dead"
        )
        typer.echo(
            f"  predicted:    {metrics.pred_viable} viable / {metrics.pred_dead} dead"
        )
        typer.echo(
            "  confusion [[TN, FP], [FN, TP]] "
            f"(label 0=dead, 1=viable): {metrics.confusion}"
        )
