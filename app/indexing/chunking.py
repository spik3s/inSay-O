from __future__ import annotations

from typing import Iterable, Sequence

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document


def get_node_parser(chunk_size: int = 1000, chunk_overlap: int = 150) -> SentenceSplitter:
    return SentenceSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, paragraph_separator="\n\n"
    )


def chunk_documents(
    docs: Iterable[Document], chunk_size: int = 1000, chunk_overlap: int = 150
) -> Sequence[BaseNode]:
    parser = get_node_parser(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return parser.get_nodes_from_documents(list(docs))
