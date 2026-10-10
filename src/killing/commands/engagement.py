from pathlib import Path

import typer

from killing.app import app
from killing.services.engagement import run_engagement


@app.command()
def engagement(
    workspace: Path = typer.Argument(..., help="Workspace with roi/ and assay.json"),
) -> None:
    """Count fluorescent engagers into analysis/Pos{n}/engagement.csv."""
    run_engagement(workspace)
