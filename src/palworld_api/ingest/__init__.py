"""Ingest adapters: everything that turns an outside source into a `Dataset`.

Nothing in here is imported by the engine. The dependency runs one way, so a
broken or replaced source cannot affect breeding, routing or passive maths.
"""

from .base import IngestError, Source

__all__ = ["IngestError", "Source"]
