"""Scraper for a community pal database.

IMPORTANT -- the selectors in this module have NOT been validated against the
live site. The environment this was written in blocks outbound access to the
source, so the parsing strategy was chosen to be as robust as possible to a
layout that could not be inspected:

- Fields are located by their *label text* ("CombiRank", "Breeding Rank", ...)
  and read from the neighbouring cell, rather than by CSS path. A wiki that
  restyles its tables keeps working; only renaming the labels breaks it.
- Every field that cannot be found is reported, not silently defaulted. A pal
  with no rank is emitted with `combi_rank=None`, which the engine excludes
  from breeding rather than guessing at.
- `python -m palworld_api.ingest.cli inspect <url>` dumps exactly what the
  parser sees on one page, which is the fastest way to fix a selector.

Run `... cli scrape --limit 5` first and read the output before trusting a
full run.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from selectolax.parser import HTMLParser, Node

from ..models import (
    Confidence,
    Dataset,
    Element,
    Pal,
    Provenance,
    Stat,
    Work,
)
from .base import IngestError
from .http import PoliteClient

# Label aliases, lower-cased and punctuation-stripped before comparison.
# Add to these rather than editing parsing code when the site renames a field.
RANK_LABELS = ("combirank", "breedingrank", "breedrank", "breedingpower", "rank")
PALDECK_LABELS = ("paldeck", "paldeckno", "number", "no", "index")
ELEMENT_LABELS = ("element", "elements", "type", "types")

WORK_LABELS: dict[str, Work] = {
    "kindling": Work.KINDLING,
    "watering": Work.WATERING,
    "planting": Work.PLANTING,
    "generatingelectricity": Work.GENERATING_ELECTRICITY,
    "electricity": Work.GENERATING_ELECTRICITY,
    "handiwork": Work.HANDIWORK,
    "gathering": Work.GATHERING,
    "lumbering": Work.LUMBERING,
    "mining": Work.MINING,
    "medicineproduction": Work.MEDICINE_PRODUCTION,
    "medicine": Work.MEDICINE_PRODUCTION,
    "cooling": Work.COOLING,
    "transporting": Work.TRANSPORTING,
    "farming": Work.FARMING,
}

STAT_LABELS: dict[str, Stat] = {
    "attack": Stat.ATTACK,
    "meleeattack": Stat.ATTACK,
    "defense": Stat.DEFENSE,
    "defence": Stat.DEFENSE,
    "hp": Stat.HP,
    "health": Stat.HP,
    "workspeed": Stat.WORK_SPEED,
    "movementspeed": Stat.MOVEMENT_SPEED,
    "runspeed": Stat.MOVEMENT_SPEED,
}

ELEMENT_NAMES: dict[str, Element] = {e.value: e for e in Element}
ELEMENT_NAMES.update({"normal": Element.NEUTRAL, "leaf": Element.GRASS})


def normalise_label(text: str) -> str:
    """Strip case, accents, whitespace and punctuation so labels compare cleanly."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", stripped.lower())


def slugify(name: str) -> str:
    """Turn a display name into a stable id."""
    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "_", stripped.lower()).strip("_")
    if not slug:
        raise IngestError(f"cannot build an id from name {name!r}")
    return slug


def first_int(text: str) -> int | None:
    match = re.search(r"-?\d+", text.replace(",", ""))
    return int(match.group()) if match else None


@dataclass
class ParsedPal:
    """What one page yielded, plus what it failed to yield.

    `missing` is the important half: it is what turns a quiet mis-parse into a
    visible data-quality report.
    """

    pal: Pal
    missing: tuple[str, ...] = ()
    source_url: str = ""


@dataclass
class PaldbSource:
    """Builds a `Dataset` from a community pal database.

    `base_url` is a parameter rather than a constant so the adapter can be
    pointed at a mirror, a local snapshot, or a different wiki with the same
    shape, without editing code.
    """

    base_url: str
    cache_dir: Path
    name: str = "paldb"
    game_version: str | None = None
    min_delay: float = 1.5
    limit: int | None = None
    respect_robots: bool = True
    index_path: str = "/en/Pals"

    _warnings: list[str] = field(default_factory=list, init=False)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def fetch(self) -> Dataset:
        retrieved_at = datetime.now(UTC).isoformat()
        with PoliteClient(
            cache_dir=self.cache_dir,
            min_delay=self.min_delay,
            respect_robots=self.respect_robots,
        ) as client:
            index_html = client.get(self.base_url.rstrip("/") + self.index_path)
            links = self.parse_index(index_html)
            if not links:
                raise IngestError(
                    f"no pal links found on the index page. The site layout has "
                    f"probably changed; run `cli inspect "
                    f"{self.base_url.rstrip('/') + self.index_path}` to see what "
                    f"the parser is looking at."
                )

            if self.limit is not None:
                links = links[: self.limit]

            parsed: list[ParsedPal] = []
            for url in links:
                html = client.get(url)
                try:
                    parsed.append(self.parse_pal(html, url, retrieved_at))
                except IngestError as exc:
                    self._warnings.append(f"{url}: {exc}")

        if not parsed:
            raise IngestError("every pal page failed to parse; refusing to emit an empty dataset")

        for item in parsed:
            if item.missing:
                self._warnings.append(
                    f"{item.pal.id}: missing {', '.join(item.missing)}"
                )

        return Dataset(
            game_version=self.game_version,
            generated_at=retrieved_at,
            source=self.name,
            pals=tuple(item.pal for item in parsed),
            # Passives and special combos come from their own pages; they are
            # merged in by the CLI rather than guessed at here.
            passives=(),
            special_combos=(),
        )

    def parse_index(self, html: str) -> list[str]:
        """Every pal detail URL linked from the index page."""
        tree = HTMLParser(html)
        seen: dict[str, None] = {}
        for node in tree.css("a[href]"):
            href = node.attributes.get("href") or ""
            if not self._looks_like_pal_link(href):
                continue
            seen.setdefault(self._absolute(href), None)
        return list(seen)

    def _looks_like_pal_link(self, href: str) -> bool:
        # Detail pages sit directly under the language prefix and carry no
        # query string; list, category and asset links do not match.
        if href.startswith(("#", "mailto:", "javascript:")):
            return False
        if any(part in href for part in ("?", "/category/", "/Category:", ".png", ".jpg")):
            return False
        return bool(re.match(r"^(?:https?://[^/]+)?/[a-z]{2}/[^/]+/?$", href))

    def _absolute(self, href: str) -> str:
        if href.startswith("http"):
            return href
        return self.base_url.rstrip("/") + "/" + href.lstrip("/")

    def parse_pal(self, html: str, url: str, retrieved_at: str) -> ParsedPal:
        """Parse one pal detail page."""
        tree = HTMLParser(html)
        name = self._parse_name(tree)
        if not name:
            raise IngestError("could not find a pal name on the page")

        pairs = self._label_value_pairs(tree)
        missing: list[str] = []

        rank = self._lookup_int(pairs, RANK_LABELS)
        if rank is None:
            missing.append("combi_rank")

        paldeck_raw = self._lookup_text(pairs, PALDECK_LABELS)
        paldeck = paldeck_raw.strip() if paldeck_raw else None
        if paldeck is None:
            missing.append("paldeck")

        elements = self._parse_elements(pairs)
        if not elements:
            missing.append("elements")

        work = self._parse_work(pairs)
        if not work:
            missing.append("work")

        stats = self._parse_stats(pairs)

        provenance = Provenance(
            source=self.name,
            confidence=Confidence.WIKI,
            url=url,
            retrieved_at=retrieved_at,
            game_version=self.game_version,
        )
        pal = Pal(
            id=slugify(name),
            name=name,
            paldeck=paldeck,
            elements=elements,
            combi_rank=rank,
            work=work,
            stats=stats,
            provenance=provenance,
        )
        return ParsedPal(pal=pal, missing=tuple(missing), source_url=url)

    @staticmethod
    def _parse_name(tree: HTMLParser) -> str | None:
        for selector in ("h1", "h2", "title"):
            node = tree.css_first(selector)
            if node is None:
                continue
            text = node.text(strip=True)
            if text:
                # Page titles often carry a site suffix; keep the leading part.
                return re.split(r"\s*[|–—-]\s*", text)[0].strip()
        return None

    def _label_value_pairs(self, tree: HTMLParser) -> dict[str, list[str]]:
        """Every label -> value pair the page exposes, however it is marked up.

        Covers table rows, definition lists, and the common
        `<div><span>Label</span><span>Value</span></div>` pattern. Collecting
        them all up front means the field parsers never care which of the three
        the site happens to use.
        """
        pairs: dict[str, list[str]] = {}

        def record(label_node: Node, value_nodes: list[Node]) -> None:
            key = normalise_label(label_node.text(strip=True))
            if not key:
                return
            values = [n.text(separator=" ", strip=True) for n in value_nodes]
            values = [v for v in values if v]
            if values:
                pairs.setdefault(key, []).extend(values)

        for row in tree.css("tr"):
            cells = row.css("th, td")
            if len(cells) >= 2:
                record(cells[0], list(cells[1:]))

        for definition_list in tree.css("dl"):
            children = [
                n for n in definition_list.iter() if n.tag in ("dt", "dd")
            ]
            current: Node | None = None
            for node in children:
                if node.tag == "dt":
                    current = node
                elif current is not None:
                    record(current, [node])

        for container in tree.css("div, li, p"):
            children = [n for n in container.iter() if n.tag in ("span", "b", "strong")]
            if len(children) == 2:
                record(children[0], [children[1]])

        return pairs

    @staticmethod
    def _lookup_text(pairs: dict[str, list[str]], labels: tuple[str, ...]) -> str | None:
        for label in labels:
            values = pairs.get(label)
            if values:
                return values[0]
        return None

    def _lookup_int(
        self, pairs: dict[str, list[str]], labels: tuple[str, ...]
    ) -> int | None:
        text = self._lookup_text(pairs, labels)
        return first_int(text) if text else None

    def _parse_elements(self, pairs: dict[str, list[str]]) -> tuple[Element, ...]:
        text = self._lookup_text(pairs, ELEMENT_LABELS)
        if not text:
            return ()
        found: list[Element] = []
        for token in re.split(r"[\s,/|]+", text):
            element = ELEMENT_NAMES.get(normalise_label(token))
            if element is not None and element not in found:
                found.append(element)
        return tuple(found)

    @staticmethod
    def _parse_work(pairs: dict[str, list[str]]) -> dict[Work, int]:
        work: dict[Work, int] = {}
        for label, values in pairs.items():
            suitability = WORK_LABELS.get(label)
            if suitability is None:
                continue
            level = first_int(values[0])
            if level is not None and 1 <= level <= 6:
                work[suitability] = level
        return work

    @staticmethod
    def _parse_stats(pairs: dict[str, list[str]]) -> dict[Stat, float]:
        stats: dict[Stat, float] = {}
        for label, values in pairs.items():
            stat = STAT_LABELS.get(label)
            if stat is None:
                continue
            value = first_int(values[0])
            if value is not None:
                stats[stat] = float(value)
        return stats
