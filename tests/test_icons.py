"""Tests for the pal icon sync.

Built the same way as test_gamefiles.py: a synthetic extraction directory, no
network, no real image bytes needed -- these tests only care about which
files get matched to which pal id and copied where.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from palworld_api.ingest.base import IngestError
from palworld_api.ingest.icons import existing_icon_ids, sync_icons
from palworld_api.models import Confidence, Dataset, Pal, Provenance

SEED = Provenance(source="test", confidence=Confidence.SEED)


def _pal(pal_id: str) -> Pal:
    return Pal(id=pal_id, name=pal_id.title(), combi_rank=1, provenance=SEED)


@pytest.fixture
def dataset() -> Dataset:
    return Dataset(
        pals=(
            _pal("anubis"),
            _pal("sheepball"),
            _pal("catmage_fire"),
            # In the roster, but the extraction below has no icon for it.
            _pal("nolcorn"),
        )
    )


@pytest.fixture
def extraction_root(tmp_path: Path) -> Path:
    icon_dir = tmp_path / "Pal" / "Content" / "Pal" / "Texture" / "PalIcon" / "Normal"
    icon_dir.mkdir(parents=True)
    for filename in (
        "T_Anubis_icon_normal.png",
        "T_SheepBall_icon_normal.png",
        "T_CatMage_Fire_icon_normal.png",
        # Belongs to a tribe not in the roster; must be ignored, not errored.
        "T_SomeBossVariant_icon_normal.png",
        # Does not match the naming pattern at all.
        "readme.txt",
    ):
        (icon_dir / filename).write_bytes(b"fake-png-bytes")
    return tmp_path


def test_copies_every_matching_icon(extraction_root: Path, dataset: Dataset, tmp_path: Path) -> None:
    output = tmp_path / "out"
    report = sync_icons(extraction_root, output, dataset)
    assert set(report.copied) == {"anubis", "sheepball", "catmage_fire"}
    assert (output / "anubis.png").read_bytes() == b"fake-png-bytes"
    assert (output / "sheepball.png").exists()
    assert (output / "catmage_fire.png").exists()


def test_reports_pals_with_no_icon(extraction_root: Path, dataset: Dataset, tmp_path: Path) -> None:
    report = sync_icons(extraction_root, tmp_path / "out", dataset)
    assert report.missing == ("nolcorn",)


def test_files_for_unknown_tribes_are_not_copied(
    extraction_root: Path, dataset: Dataset, tmp_path: Path
) -> None:
    output = tmp_path / "out"
    sync_icons(extraction_root, output, dataset)
    assert not (output / "somebossvariant.png").exists()


def test_unmatched_files_are_counted(extraction_root: Path, dataset: Dataset, tmp_path: Path) -> None:
    report = sync_icons(extraction_root, tmp_path / "out", dataset)
    # readme.txt never matches the `T_*_icon_normal.png` glob at all, so only
    # the boss variant -- a real icon file for a tribe outside the roster --
    # counts as unmatched.
    assert report.unmatched_files == 1


def test_coverage_reports_a_fraction(extraction_root: Path, dataset: Dataset, tmp_path: Path) -> None:
    report = sync_icons(extraction_root, tmp_path / "out", dataset)
    assert report.coverage == "3/4"


def test_missing_extraction_directory_is_a_clear_error(dataset: Dataset, tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="missing icon directory"):
        sync_icons(tmp_path / "nope", tmp_path / "out", dataset)


def test_existing_icon_ids_reads_the_output_directory(tmp_path: Path) -> None:
    (tmp_path / "anubis.png").write_bytes(b"x")
    (tmp_path / "sheepball.png").write_bytes(b"x")
    (tmp_path / "notes.txt").write_bytes(b"x")
    assert existing_icon_ids(tmp_path) == frozenset({"anubis", "sheepball"})


def test_existing_icon_ids_is_empty_for_a_missing_directory(tmp_path: Path) -> None:
    assert existing_icon_ids(tmp_path / "nope") == frozenset()


def test_sync_is_idempotent(extraction_root: Path, dataset: Dataset, tmp_path: Path) -> None:
    output = tmp_path / "out"
    first = sync_icons(extraction_root, output, dataset)
    second = sync_icons(extraction_root, output, dataset)
    assert first.copied == second.copied
