from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import Document

from app.models.metadata import SourceType, file_metadata

TEXT_EXTS = {".md", ".txt"}
CSV_EXTS = {".csv"}


def load_text_documents(input_dir: Path) -> List[Document]:
    reader = SimpleDirectoryReader(
        input_dir=str(input_dir),
        recursive=True,
        required_exts=list(TEXT_EXTS),
        file_metadata=lambda p: file_metadata(Path(p), source=SourceType.local_file),
    )
    return reader.load_data()


def load_csv_documents(input_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for path in input_dir.rglob("*.csv"):
        meta = file_metadata(
            path, source=SourceType.jira_csv
        )  # TODO: Why is this a jira_CSV source type? We might want to use different kinds of csv
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        header, body = (rows[0], rows[1:]) if rows else ([], [])
        text_lines = ["CSV File:", str(path), "", "Header:", ", ".join(header), "", "Rows:"]
        preview = body[:200]
        for r in preview:
            text_lines.append(", ".join(r))
        if len(body) > len(preview):
            text_lines.append(f"... ({len(body) - len(preview)} more rows)")
        text = "\n".join(text_lines)
        docs.append(Document(text=text, metadata=meta))
    return docs


def load_local_documents(input_dir: str | Path) -> List[Document]:
    base = Path(input_dir)
    if not base.exists():
        return []
    docs = load_text_documents(base)
    docs += load_csv_documents(base)
    return docs
