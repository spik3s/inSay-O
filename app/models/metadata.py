from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any


class SourceType(StrEnum):
    local_file = "local_file"
    jira_csv = "jira_csv"
    github = "github"


def file_metadata(
    path: Path, source: SourceType = SourceType.local_file, **extra: Any
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "source": source.value,
        "path": str(path),
        "name": path.name,
        "ext": path.suffix.lower(),
        "size": stat.st_size,
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        **extra,
    }
