from pathlib import Path

import typer

from killing.app import app
from killing.services.classifier import run_clean


@app.command()
def clean(
    workspace: Path = typer.Argument(
        ..., help="Workspace with results/predictions.csv"
    ),
) -> None:
    """Clean classifier labels and write death times and the kill curve."""
    run_clean(workspace)
