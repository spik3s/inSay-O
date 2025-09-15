from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from llama_index.core.schema import BaseNode
from rank_bm25 import BM25Okapi

TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if len(t) > 1]


@dataclass
class BM25Record:
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class BM25Store:
    records: List[BM25Record]
    tokens: List[List[str]]
    # BM25Okapi is reconstructed on load for portability

    def build_bm25(self) -> BM25Okapi:
        return BM25Okapi(self.tokens)


def build_store_from_nodes(nodes: Sequence[BaseNode]) -> BM25Store:
    records: List[BM25Record] = []
    tokens: List[List[str]] = []
    for n in nodes:
        # n is a TextNode from LlamaIndex
        text = n.get_content()
        rec = BM25Record(id=n.node_id, text=text, metadata=dict(n.metadata))
        records.append(rec)
        tokens.append(tokenize(text))
    return BM25Store(records=records, tokens=tokens)


def save_store(store: BM25Store, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:  # wb stands for write binary
        pickle.dump(store, f)


def load_store(path: str | Path) -> BM25Store:
    with Path(path).open("rb") as f:  # and rb for read binary
        return pickle.load(f)


def search(store: BM25Store, query: str, k: int = 40) -> List[Tuple[int, float]]:
    bm25 = store.build_bm25()
    scores = bm25.get_scores(tokenize(query))
    pairs: List[Tuple[int, float]] = [(i, float(s)) for i, s in enumerate(scores)]
    pairs.sort(key=lambda t: t[1], reverse=True)
    return pairs[: max(0, k)]
