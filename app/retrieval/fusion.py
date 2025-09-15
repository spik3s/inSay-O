from __future__ import annotations

from typing import Any, Dict, List


def rrf_fuse(
    dense: List[Dict[str, Any]], sparse: List[Dict[str, Any]], k: int = 60, limit: int = 50
) -> List[Dict[str, Any]]:
    # dense/sparse are lists of candidates with unique 'id' and descending 'score'
    ranks: dict[str, float] = {}
    info: dict[str, Dict[str, Any]] = {}

    for lst in (dense, sparse):
        for rank, c in enumerate(lst, start=1):
            cid = c["id"]
            info[cid] = c  # preserve latest metadata/text seen
            ranks[cid] = ranks.get(cid, 0.0) + 1.0 / (k + rank)

    fused = [{**info[cid], "rrf": score} for cid, score in ranks.items()]
    fused.sort(key=lambda x: x["rrf"], reverse=True)
    return fused[:limit]
