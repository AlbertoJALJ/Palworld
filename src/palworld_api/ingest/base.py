"""The contract every ingest adapter implements."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..models import Dataset


class IngestError(RuntimeError):
    """Raised when a source cannot be turned into a valid dataset."""


@runtime_checkable
class Source(Protocol):
    """A place a `Dataset` can be built from.

    Adapters must either return a dataset that validates or raise
    `IngestError`. Returning a half-populated dataset is not an option: a
    silently incomplete pal list produces confidently wrong breeding routes,
    which is worse than no answer at all.
    """

    name: str

    def fetch(self) -> Dataset:
        """Build a complete, validated dataset."""
        ...
