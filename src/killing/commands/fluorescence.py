from pathlib import Path

import typer

from killing.app import app
from killing.services.fluorescence import run_fluorescence


@app.command()
def fluorescence(
    workspace: Path = typer.Argument(..., help="Workspace with roi/ and assay.json"),
) -> None:
    """Measure death-reporter fluorescence into analysis/Pos{n}/ch{m}.csv."""
    run_fluorescence(workspace)
