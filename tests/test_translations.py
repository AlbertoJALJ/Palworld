"""Tests for the translation sidecar loader."""

from __future__ import annotations

import json
from pathlib import Path

from palworld_api.translations import Translations, load_translations


def test_missing_file_returns_none(tmp_path: Path) -> None:
    assert load_translations(tmp_path / "nope.json") is None


def test_unreadable_json_returns_none(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert load_translations(path) is None


def test_loads_pals_and_passives(tmp_path: Path) -> None:
    path = tmp_path / "es.json"
    path.write_text(
        json.dumps({"locale": "es", "pals": {"sheepball": "Lamball"}, "passives": {"legend": "Leyenda"}}),
        encoding="utf-8",
    )
    translation = load_translations(path)
    assert translation is not None
    assert translation.locale == "es"
    assert translation.pal_name("sheepball", "fallback") == "Lamball"
    assert translation.passive_name("legend", "fallback") == "Leyenda"


def test_missing_entry_falls_back(tmp_path: Path) -> None:
    path = tmp_path / "es.json"
    path.write_text(json.dumps({"locale": "es", "pals": {}, "passives": {}}), encoding="utf-8")
    translation = load_translations(path)
    assert translation is not None
    assert translation.pal_name("unknown_id", "English Name") == "English Name"
    assert translation.passive_name("unknown_id", "English Name") == "English Name"


def test_locale_defaults_to_filename_stem_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "es.json"
    path.write_text(json.dumps({"pals": {}, "passives": {}}), encoding="utf-8")
    translation = load_translations(path)
    assert translation is not None
    assert translation.locale == "es"


def test_translations_object_defaults_are_empty() -> None:
    translation = Translations(locale="es")
    assert translation.pal_name("x", "Fallback") == "Fallback"
    assert translation.passive_name("x", "Fallback") == "Fallback"
