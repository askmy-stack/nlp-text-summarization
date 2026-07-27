"""Path sandboxing for server-side file reads in the serving layer.

The multimodal JSON endpoint lets a caller pass a server-side ``path``
instead of uploading bytes directly. Without restriction, that path is
handed straight to ``PIL.Image.open`` / ``ffmpeg`` / the Whisper pipeline,
which is an arbitrary file read (e.g. ``path="/etc/passwd"`` or
``path="../../../etc/passwd"``). This module confines any such path to an
allowlisted sandbox directory before it is used.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import HTTPException

DEFAULT_SANDBOX_DIR = "artifacts/multimodal_uploads"


def get_sandbox_dir() -> Path:
    """Return the allowlisted directory that multimodal ``path`` values must resolve within.

    Configurable via the ``MULTIMODAL_SANDBOX_DIR`` environment variable so
    deployments can point it at a dedicated upload volume.
    """
    raw = os.getenv("MULTIMODAL_SANDBOX_DIR", DEFAULT_SANDBOX_DIR)
    base = Path(raw).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base


def resolve_sandboxed_path(user_path: str, base_dir: Path | None = None) -> Path:
    """Resolve ``user_path`` and confirm it stays within the sandbox directory.

    Handles absolute-path escapes, ``..`` traversal, and symlink escapes by
    resolving to the real filesystem path and checking containment against
    the resolved sandbox root. Raises ``HTTPException(403)`` on any escape
    attempt.
    """
    base = base_dir if base_dir is not None else get_sandbox_dir()

    candidate_input = Path(user_path)
    unresolved = candidate_input if candidate_input.is_absolute() else base / candidate_input
    candidate = unresolved.resolve()

    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise HTTPException(
            status_code=403,
            detail="Path is outside the allowed sandbox directory",
        ) from exc

    return candidate
