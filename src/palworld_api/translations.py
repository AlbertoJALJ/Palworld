"""Loading the translation sidecar file produced by `ingest/i18n.py`.

Kept separate from the core dataset on purpose: `data/pals.json` is the game
data this project verifies against a second source and refuses to guess at,
while `data/i18n/es.json` is a supplementary, partial translation layer with
its own confidence level (community site, not the game's own files) and its
own honest gaps (some passive names cannot be resolved uniquely -- see
`ingest/i18n.py`). Missing entirely, this file makes the API and web UI
degrade to English rather than fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Translations:
    """One locale's known names, keyed by this project's own ids."""

    locale: str
    pals: dict[str, str] = field(default_factory=dict)
    passives: dict[str, str] = field(default_factory=dict)

    def pal_name(self, pal_id: str, fallback: str) -> str:
        return self.pals.get(pal_id, fallback)

    def passive_name(self, passive_id: str, fallback: str) -> str:
        return self.passives.get(passive_id, fallback)


def load_translations(path: str | Path) -> Translations | None:
    """Read a translation sidecar file, or None if it does not exist.

    Unlike `load_dataset`, a missing or unreadable file is not an error: it
    just means this locale is not available, and callers fall back to the
    dataset's own (English) names.
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    return Translations(
        locale=raw.get("locale", path.stem),
        pals=raw.get("pals", {}),
        passives=raw.get("passives", {}),
    )
