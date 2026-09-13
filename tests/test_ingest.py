"""Parser tests for the scraping adapter.

These pin the parser's *logic* against HTML written here. They deliberately do
not prove the selectors match the live site -- nothing offline can prove that.
What they do prove is that when a page presents its fields in any of the three
markup shapes the adapter claims to support, the right values come out, and that
a page missing a field reports it instead of inventing one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from palworld_api.ingest.base import IngestError
from palworld_api.ingest.paldb import (
    PaldbSource,
    first_int,
    normalise_label,
    slugify,
)
from palworld_api.models import Element, Stat, Work

RETRIEVED = "2026-09-13T00:00:00+00:00"


@pytest.fixture
def source(tmp_path: Path) -> PaldbSource:
    return PaldbSource(base_url="https://example.test", cache_dir=tmp_path)


TABLE_PAGE = """
<html><head><title>Anubis | Example Wiki</title></head><body>
  <h1>Anubis</h1>
  <table>
    <tr><th>Paldeck</th><td>099</td></tr>
    <tr><th>Element</th><td>Ground</td></tr>
    <tr><th>CombiRank</th><td>1,340</td></tr>
    <tr><th>Handiwork</th><td>Lv4</td></tr>
    <tr><th>Mining</th><td>Lv3</td></tr>
    <tr><th>Attack</th><td>120</td></tr>
    <tr><th>Defense</th><td>100</td></tr>
  </table>
</body></html>
"""

DEFINITION_LIST_PAGE = """
<html><body>
  <h1>Depresso</h1>
  <dl>
    <dt>Paldeck No.</dt><dd>045</dd>
    <dt>Type</dt><dd>Dark</dd>
    <dt>Breeding Rank</dt><dd>820</dd>
    <dt>Transporting</dt><dd>Lv2</dd>
  </dl>
</body></html>
"""

SPAN_PAIR_PAGE = """
<html><body>
  <h1>Lamball</h1>
  <div><span>Paldeck</span><span>001</span></div>
  <div><span>Elements</span><span>Neutral</span></div>
  <div><span>CombiRank</span><span>1470</span></div>
  <div><span>Farming</span><span>Lv1</span></div>
</body></html>
"""


def test_parses_a_table_layout(source: PaldbSource) -> None:
    parsed = source.parse_pal(TABLE_PAGE, "https://example.test/en/Anubis", RETRIEVED)
    pal = parsed.pal
    assert pal.id == "anubis"
    assert pal.name == "Anubis"
    assert pal.paldeck == "099"
    assert pal.combi_rank == 1340  # thousands separator must not truncate it
    assert pal.elements == (Element.GROUND,)
    assert pal.work == {Work.HANDIWORK: 4, Work.MINING: 3}
    assert pal.stats == {Stat.ATTACK: 120.0, Stat.DEFENSE: 100.0}
    assert parsed.missing == ()


def test_parses_a_definition_list_layout(source: PaldbSource) -> None:
    pal = source.parse_pal(DEFINITION_LIST_PAGE, "u", RETRIEVED).pal
    assert pal.name == "Depresso"
    assert pal.paldeck == "045"
    assert pal.combi_rank == 820
    assert pal.elements == (Element.DARK,)
    assert pal.work == {Work.TRANSPORTING: 2}


def test_parses_a_span_pair_layout(source: PaldbSource) -> None:
    pal = source.parse_pal(SPAN_PAIR_PAGE, "u", RETRIEVED).pal
    assert pal.name == "Lamball"
    assert pal.combi_rank == 1470
    assert pal.elements == (Element.NEUTRAL,)
    assert pal.work == {Work.FARMING: 1}


def test_missing_fields_are_reported_not_invented(source: PaldbSource) -> None:
    page = "<html><body><h1>Mystery</h1><table><tr><th>Nonsense</th><td>1</td></tr></table></body></html>"
    parsed = source.parse_pal(page, "u", RETRIEVED)
    assert parsed.pal.combi_rank is None
    assert "combi_rank" in parsed.missing
    assert "elements" in parsed.missing
    assert "work" in parsed.missing


def test_a_pal_with_no_rank_is_excluded_from_breeding(source: PaldbSource) -> None:
    page = "<html><body><h1>Mystery</h1></body></html>"
    pal = source.parse_pal(page, "u", RETRIEVED).pal
    assert pal.is_breedable_parent is False
    assert pal.is_breedable_child is False


def test_a_page_with_no_name_is_an_error(source: PaldbSource) -> None:
    with pytest.raises(IngestError, match="could not find a pal name"):
        source.parse_pal("<html><body><table></table></body></html>", "u", RETRIEVED)


def test_multiple_elements_are_all_captured(source: PaldbSource) -> None:
    page = """<html><body><h1>Hybrid</h1><table>
      <tr><th>Elements</th><td>Fire / Dragon</td></tr></table></body></html>"""
    pal = source.parse_pal(page, "u", RETRIEVED).pal
    assert pal.elements == (Element.FIRE, Element.DRAGON)


def test_provenance_records_where_the_value_came_from(source: PaldbSource) -> None:
    pal = source.parse_pal(TABLE_PAGE, "https://example.test/en/Anubis", RETRIEVED).pal
    assert pal.provenance is not None
    assert pal.provenance.url == "https://example.test/en/Anubis"
    assert pal.provenance.retrieved_at == RETRIEVED
    assert pal.provenance.verified is True


def test_index_parsing_keeps_only_detail_links(source: PaldbSource) -> None:
    index = """<html><body>
      <a href="/en/Anubis">Anubis</a>
      <a href="/en/Lamball">Lamball</a>
      <a href="/en/Anubis">Anubis again</a>
      <a href="/en/Pals?page=2">Next</a>
      <a href="/en/category/Ground">Category</a>
      <a href="/images/anubis.png">Image</a>
      <a href="#top">Top</a>
    </body></html>"""
    links = source.parse_index(index)
    assert links == [
        "https://example.test/en/Anubis",
        "https://example.test/en/Lamball",
    ]


def test_absolute_links_are_preserved(source: PaldbSource) -> None:
    index = '<a href="https://example.test/en/Anubis">A</a>'
    assert source.parse_index(index) == ["https://example.test/en/Anubis"]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CombiRank", "combirank"),
        ("Combi Rank", "combirank"),
        ("  Paldeck No. ", "paldeckno"),
        ("Générateur", "generateur"),
    ],
)
def test_label_normalisation(raw: str, expected: str) -> None:
    assert normalise_label(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Anubis", "anubis"), ("Relaxaurus Lux", "relaxaurus_lux"), ("Jelly-Fin", "jelly_fin")],
)
def test_slugify(raw: str, expected: str) -> None:
    assert slugify(raw) == expected


def test_slugify_rejects_unusable_names() -> None:
    with pytest.raises(IngestError):
        slugify("!!!")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Lv4", 4), ("1,340", 1340), ("-5", -5), ("none", None), ("", None)],
)
def test_first_int(raw: str, expected: int | None) -> None:
    assert first_int(raw) == expected
