from __future__ import annotations

import shutil
from pathlib import Path

import typer

from app.config import get_settings

from .ingest import main as ingest_main  # reuse ingest

app = typer.Typer(add_completion=False)


@app.command()
def main(
    input_dir: str = typer.Option("data", help="Directory of docs to ingest"),
    confirm: bool = typer.Option(True, help="Require confirmation before deleting index"),
) -> None:
    s = get_settings()
    chroma_path = Path(s.CHROMA_PERSIST_DIR)
    if confirm:
        typer.confirm(f"This will delete {chroma_path}. Continue?", abort=True)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)
    ingest_main.callback = None  # silence Typer warning in reuse
    ingest_main(input_dir=input_dir)


if __name__ == "__main__":
    app()
