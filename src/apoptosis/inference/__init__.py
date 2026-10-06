"""Label-free viability model and classifier for the LiSCA inference host."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def register(
    register_model: Callable[[str, Any], None], register_task: Callable[..., None]
) -> None:
    """Register EmbeddingGemma and the viable/dead task. Called by lisca.inference."""
    from apoptosis.inference.embeddinggemma import EmbeddingGemmaEncoder
    from apoptosis.inference.viability import mount, prepare

    register_model("embeddinggemma-2", EmbeddingGemmaEncoder)
    register_task(prepare, mount)
