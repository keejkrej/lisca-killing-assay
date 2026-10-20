from pathlib import Path

import typer

from killing.app import app
from killing.core.session import DEFAULT_DATA_DIR, PROJECT_ROOT
from killing.services.head_train import default_head_path
from killing.services.inference import infer_all_cells, save_inference
from killing.services.plot_validation import plot_validation


@app.command()
def predict(
    data_dir: Path = typer.Option(
        DEFAULT_DATA_DIR,
        "--data-dir",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    head: Path = typer.Option(
        default_head_path(PROJECT_ROOT),
        "--head",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_json: Path = typer.Option(
        PROJECT_ROOT / "runs" / "viability" / "inference.json",
        "--output-json",
        resolve_path=True,
    ),
    output_plot: Path = typer.Option(
        PROJECT_ROOT / "runs" / "viability" / "validation_plot.png",
        "--output-plot",
        resolve_path=True,
    ),
    batch_size: int = typer.Option(8, min=1),
    device: str = typer.Option("cuda", help="cuda or cpu for EmbeddingGemma"),
) -> None:
    """Score every ROI with EmbeddingGemma and the trained linear head."""
    typer.echo(f"Inferring {data_dir} with {head.name}")
    results = infer_all_cells(
        data_dir=data_dir,
        head_path=head,
        batch_size=batch_size,
        device=device,
    )
    save_inference(results, output_json)
    plot_validation(results, output_plot)
    typer.echo(f"Wrote {len(results)} cell predictions to {output_json}")
    typer.echo(f"Wrote validation figure to {output_plot}")
