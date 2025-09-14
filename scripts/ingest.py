from __future__ import annotations

import typer

from app.config import get_settings
from app.indexing.chunking import chunk_documents
from app.indexing.vector_store import index_nodes
from app.ingestion.local import load_local_documents

app = typer.Typer(add_completion=False)


@app.command()
def main(input_dir: str = typer.Option("data", help="Directory of docs to ingest")) -> None:
    s = get_settings()
    docs = load_local_documents(input_dir)
    if not docs:
        typer.secho("No documents found to ingest.", fg=typer.colors.YELLOW)
        raise typer.Exit(code=1)
    nodes = chunk_documents(docs)
    index_nodes(nodes, persist_dir=s.CHROMA_PERSIST_DIR, embedding_model=s.EMBEDDING_MODEL)
    typer.secho(f"Ingested {len(nodes)} chunks into collection.", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
