from __future__ import annotations

import typer

from app.config import get_settings
from app.indexing.vector_store import load_index

app = typer.Typer(add_completion=False)


@app.command()
def main(
    query: str = typer.Argument(..., help="Query text"),
    top_k: int = typer.Option(8, "--top-k", "-k", help="Top-k results"),
) -> None:
    s = get_settings()
    index = load_index(persist_dir=s.CHROMA_PERSIST_DIR, embedding_model=s.EMBEDDING_MODEL)
    qe = index.as_query_engine(similarity_top_k=top_k)
    resp = qe.query(query)
    print("\n=== ANSWER (draft, no LLM synthesis) ===\n")
    print(getattr(resp, "response", "(response text not available)"))
    print("\n=== CONTEXT CHUNKS ===\n")
    for i, sn in enumerate(getattr(resp, "source_nodes", [])[:top_k], start=1):
        meta = sn.node.metadata if hasattr(sn, "node") else {}
        text = sn.node.get_content()[:300].replace("\n", " ") if hasattr(sn, "node") else ""
        print(
            f"[{i}] score={sn.score:.3f} "
            f"source={meta.get('source')} "
            f"path={meta.get('path')}\n{text}\n"
        )


if __name__ == "__main__":
    app()
