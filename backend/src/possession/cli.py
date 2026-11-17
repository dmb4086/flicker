"""Command line: `possession submit <data-dir> --out <team>.csv`."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def main() -> None:
    """Frame-level football possession from automatic tracking."""


@app.command()
def submit(data_dir: Annotated[Path, typer.Argument(help="the competition data/ folder")],
           out: Annotated[Path, typer.Option(help="submission CSV to write")],
           work_dir: Annotated[Path | None, typer.Option(help="scratch folder")] = None,
           pathcrf_python: Annotated[Path | None, typer.Option(
               help="python of the path model's environment")] = None) -> None:
    """Data folder -> frame,label CSV. Writes <out> and <out>.manifest.json. CPU, deterministic."""
    from possession.submit import SubmitSettings, run_submit

    settings = SubmitSettings()
    if pathcrf_python is not None:
        settings = replace(settings, pathcrf_python=pathcrf_python)
    result = run_submit(data_dir.resolve(), out.resolve(), settings,
                        work_dir.resolve() if work_dir is not None else None)
    typer.echo(json.dumps(result.summary, indent=2, sort_keys=True))
    typer.echo(f"wrote {result.csv}")
    if result.summary["official_validator"] != "PASS":
        raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()
