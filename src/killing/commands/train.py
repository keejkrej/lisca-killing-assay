from pathlib import Path

import typer

from killing.app import app
from killing.core.session import PROJECT_ROOT
from killing.services.dataset_build import default_dataset_dir
from killing.services.head_train import default_head_dir, train_linear_head


@app.command()
def train(
    manifest_path: Path = typer.Option(
        default_dataset_dir(PROJECT_ROOT) / "manifest.json",
        "--manifest",
        exists=True,
        dir_okay=False,
        resolve_path=True,
    ),
    output_dir: Path = typer.Option(
        default_head_dir(PROJECT_ROOT),
        "--output-dir",
        resolve_path=True,
    ),
    embeddings_path: Path | None = typer.Option(
        None,
        "--embeddings",
        help="Optional cached embeddings (.npz with ids and vectors).",
        dir_okay=False,
        resolve_path=True,
    ),
    svm_c: float = typer.Option(1.0, min=1e-6),
    batch_size: int = typer.Option(8, min=1),
    device: str = typer.Option("cuda", help="cuda or cpu for EmbeddingGemma"),
) -> None:
    """Train a linear SVM head on frozen EmbeddingGemma embeddings."""
    head = train_linear_head(
        manifest_path=manifest_path,
        output_dir=output_dir,
        embeddings_path=embeddings_path,
        svm_c=svm_c,
        batch_size=batch_size,
        device=device,
    )
    typer.echo(f"Encoder: {head.encoder_model_id}@{head.encoder_revision}")
    typer.echo(f"Classes: {', '.join(head.classes)}")
    typer.echo(f"Wrote {output_dir / 'head.json'}")
