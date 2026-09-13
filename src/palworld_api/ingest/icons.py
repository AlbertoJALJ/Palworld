"""Ingest of pal icon images from the game's own texture extraction.

Same shape and same philosophy as `gamefiles.py`: read straight out of an
extraction root rather than a wiki, so an icon set can be regenerated for a
new patch with one command instead of re-scraping.

Expects the icon textures at:

    <root>/Pal/Content/Pal/Texture/PalIcon/Normal/T_<TribeId>_icon_normal.png

`<TribeId>` is the same raw monster-table key `gamefiles.slug()` already
normalises into a pal id (`T_SheepBall_icon_normal.png` is Lamball's icon,
because Lamball's internal id is `sheepball`), so matching a file to a pal
reuses that one function rather than inventing a second naming rule.

Licensing note (see also the one in gamefiles.py and the README): these are
Pocketpair's own icon textures, extracted rather than drawn. Redistributing
them is normal practice across Palworld fan tooling, and this project is
non-commercial and open source, but that is a mitigating factor, not a
license -- it's the user's call whether to keep them committed. They are
regenerable from any extraction in one command, so removing them costs
nothing but re-running `icons`.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..models import Dataset
from .base import IngestError
from .gamefiles import slug

ICON_FILENAME = re.compile(r"^T_(?P<tribe>.+)_icon_normal\.png$")


@dataclass(frozen=True, slots=True)
class IconSyncReport:
    """What a sync actually did, so a partial icon set is visible, not silent."""

    copied: tuple[str, ...]
    missing: tuple[str, ...]
    unmatched_files: int

    @property
    def coverage(self) -> str:
        total = len(self.copied) + len(self.missing)
        return f"{len(self.copied)}/{total}" if total else "0/0"


def _icon_dir(root: Path) -> Path:
    return Path(root) / "Pal" / "Content" / "Pal" / "Texture" / "PalIcon" / "Normal"


def sync_icons(root: Path, output_dir: Path, dataset: Dataset) -> IconSyncReport:
    """Copy every icon the extraction has for a pal in `dataset` into `output_dir`.

    Pals the extraction has no icon for are reported, not defaulted to some
    placeholder file on disk -- the frontend decides how to show a gap, this
    layer only says where one is.
    """
    source_dir = _icon_dir(root)
    if not source_dir.is_dir():
        raise IngestError(
            f"missing icon directory: {source_dir}. Is `root` an extraction "
            f"containing Pal/Content/... ?"
        )

    # Index every icon file by the pal id it belongs to. A tribe id that does
    # not slug to a known pal (a boss/summon variant with its own icon, cut
    # content, ...) is simply not this dataset's concern.
    known_ids = {pal.id for pal in dataset.pals}
    by_pal_id: dict[str, Path] = {}
    unmatched = 0
    for file in source_dir.glob("T_*_icon_normal.png"):
        match = ICON_FILENAME.match(file.name)
        if not match:
            unmatched += 1
            continue
        pal_id = slug(match.group("tribe"))
        if pal_id in known_ids:
            by_pal_id[pal_id] = file
        else:
            unmatched += 1

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    for pal_id, source_file in sorted(by_pal_id.items()):
        shutil.copyfile(source_file, output_dir / f"{pal_id}.png")
        copied.append(pal_id)

    missing = tuple(sorted(known_ids - by_pal_id.keys()))
    return IconSyncReport(
        copied=tuple(copied), missing=missing, unmatched_files=unmatched
    )


def existing_icon_ids(icons_dir: Path) -> frozenset[str]:
    """Pal ids that currently have an icon file on disk.

    Used at API startup so a response can say which pals have a real icon
    without the frontend having to probe each one with a failed image load.
    """
    icons_dir = Path(icons_dir)
    if not icons_dir.is_dir():
        return frozenset()
    return frozenset(f.stem for f in icons_dir.glob("*.png"))
