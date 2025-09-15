from __future__ import annotations

from typing import Any, Dict, List

from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None


def get_reranker(model_name: str) -> CrossEncoder:
    global _model
    if _model is None or _model.model_name != model_name:
        _model = CrossEncoder(model_name, max_length=512)
    return _model


def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str,
    top_n: int = 10,
    batch_size: int = 32,
) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    ce = get_reranker(model_name)
    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(
        pairs, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False
    )
    for c, s in zip(candidates, scores, strict=False):
        c["ce_score"] = float(s)
    candidates.sort(key=lambda x: x["ce_score"], reverse=True)
    return candidates[:top_n]
