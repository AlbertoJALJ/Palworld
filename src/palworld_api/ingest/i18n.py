"""Spanish translations for pal and passive names.

Neither the game's shipped DataTables nor `gamefiles.py`'s source extraction
include non-English locales (see the README's Localisation section), so this
adapter reads from a community database site instead: palworld.gg, which
publishes a compiled per-pal database in each locale it supports.

Two different things are extracted, by two different methods, because the
site exposes them two different ways:

- **Pal names** come from the site's own compiled pal database, embedded as a
  JS module and matched to a pal 1:1 by the same internal tribe id this
  project already uses (`sheepball` is Lamball in both places). This is an
  exact join, same confidence as the icon match in `icons.py`.

- **Passive names** are matched indirectly, because the site's Spanish name is
  the only Spanish text available and there is no shared internal id to join
  on. Instead, each Spanish passive-skill card's rendered description is
  parsed back into an effect signature (stat + value), and matched against
  this project's own `Passive.effects`. Where more than one English-named
  passive shares the exact same numeric signature (a generic tiered skill and
  a flavour-named one worth the same amount, say), the match is ambiguous and
  is left untranslated rather than guessed -- consistent with this project's
  rule that a gap is reported, never papered over.

Because the join for passives is indirect, this file separates the parsing
logic (pure functions, fully unit tested against synthetic input) from the
network fetch (`PalworldGGSource`, which needs a real browser because the
site's passive-skill catalog paginates client-side with no plain HTTP
equivalent -- Playwright is an optional dependency for exactly this path).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from .base import IngestError

# -- pal names --------------------------------------------------------------

# Matches one compiled pal record's opening, e.g.:
#   {id:"sheepball",key:"SheepBall",slug:"lamball",name:"Lamball",...}
_PAL_RECORD_START = re.compile(r'\{id:"([a-zA-Z0-9_]+)",key:"[^"]*",slug:"[^"]*",name:"')


def _matching_brace(text: str, open_index: int) -> int:
    """Index of the `}` that closes the `{` at `open_index`, honouring strings.

    Shared by both extractors here: the compiled JS uses object and array
    literals with string values that can themselves contain `{`, `}`, `[` or
    `]` (line breaks in a description, an escaped quote), so a naive
    first-match search corrupts the boundary. This walks the text tracking
    whether it is inside a string, and which quote character opened it.
    """
    depth = 0
    in_string = False
    quote = ""
    escaped = False
    i = open_index
    while i < len(text):
        char = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                in_string = False
        else:
            if char in "\"'`":
                in_string = True
                quote = char
            elif char == text[open_index]:
                depth += 1
            elif char == {"{": "}", "[": "]"}[text[open_index]]:
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    raise IngestError(f"unbalanced {text[open_index]!r} starting at index {open_index}")


def extract_pal_names(chunk: str) -> dict[str, str]:
    """Every `pal id -> localised name` pair in a compiled pal-database chunk.

    `chunk` is the JS source of the site's pal-database module for one
    locale. Ids that do not look like this project's roster (variant or boss
    records the site tracks separately) are still returned; the caller
    intersects with its own dataset's ids.
    """
    names: dict[str, str] = {}
    for match in _PAL_RECORD_START.finditer(chunk):
        end = _matching_brace(chunk, match.start())
        record = chunk[match.start() : end + 1]
        name_match = re.search(r'name:"([^"]*)"', record)
        if name_match:
            names[match.group(1)] = name_match.group(1)
    return names


# -- passive effect parsing ---------------------------------------------------

# Spanish element names, longest/most specific first so "no elemental" is not
# shadowed by a later, shorter, unrelated match.
_ELEMENTS_ES: tuple[tuple[str, str], ...] = (
    ("no elemental", "neutral"),
    ("dragontino", "dragon"),
    ("hielo", "ice"),
    ("oscuridad", "dark"),
    ("tierra", "ground"),
    ("planta", "grass"),
    ("rayo", "electric"),
    ("agua", "water"),
    ("fuego", "fire"),
)

# Each entry: (pattern, stat, sign). `sign` is "+"/"-" to force the matched
# number's sign (the site phrases some effects as "reduced by N%", which is
# a negative value on the underlying stat), or None to keep the number's own
# sign, which the pattern must then capture explicitly.
_EFFECT_PATTERNS: tuple[tuple[str, str, str | None], ...] = (
    (r"Ataque\s*([+-]\s*\d+(?:\.\d+)?)\s*%", "attack", None),
    (r"[Aa]umenta tu ataque en un\s*(\d+(?:\.\d+)?)\s*%", "attack", "+"),
    (r"(?:el |la )?ataque en un\s*(\d+(?:\.\d+)?)\s*%", "attack", "+"),
    (r"Defensa\s*([+-]\s*\d+(?:\.\d+)?)\s*%", "defense", None),
    (r"[Aa]umenta tu defensa en un\s*(\d+(?:\.\d+)?)\s*%", "defense", "+"),
    (r"(?:el |la )?defensa en un\s*(\d+(?:\.\d+)?)\s*%", "defense", "+"),
    (r"(?:Vel\.|Velocidad) de trabajo\s*([+-]\s*\d+(?:\.\d+)?)\s*%", "work_speed", None),
    (r"[Aa]umenta tu velocidad de trabajo en un\s*(\d+(?:\.\d+)?)\s*%", "work_speed", "+"),
    (r"PV(?:\s*máximos)?\s*([+-]\s*\d+(?:\.\d+)?)\s*%", "hp", None),
    (
        r"[Aa]umenta la velocidad de (?:desplazamiento|movimiento)\s*([+-]?\s*\d+(?:\.\d+)?)"
        r"\s*%(?! en el agua)",
        "movement_speed",
        "+",
    ),
    (
        r"(?:la )?velocidad de (?:movimiento|desplazamiento) en un\s*(\d+(?:\.\d+)?)"
        r"\s*%(?! en el agua)",
        "movement_speed",
        "+",
    ),
    (r"La saciedad (?:baja|se reduce) (?:un\s*)?\+?(\d+(?:\.\d+)?)\s*%?\s*más\b", "hunger_loss", "+"),
    (r"La saciedad (?:baja|se reduce) (?:un\s*)?\+?(\d+(?:\.\d+)?)\s*%?\s*menos\b", "hunger_loss", "-"),
    (r"La saciedad se reduce con más dificultad:?\s*\+?(\d+(?:\.\d+)?)\s*%", "hunger_loss", "-"),
    (r"La saciedad se reduce más fácilmente:?\s*\+?(\d+(?:\.\d+)?)\s*%", "hunger_loss", "+"),
    (r"La COR (?:baja|se reduce) (?:un\s*)?\+?(\d+(?:\.\d+)?)\s*%?\s*más\b", "sanity_loss", "+"),
    (r"La COR (?:baja|se reduce) (?:un\s*)?\+?(\d+(?:\.\d+)?)\s*%?\s*menos\b", "sanity_loss", "-"),
    (r"La COR se reduce con más dificultad:?\s*\+?(\d+(?:\.\d+)?)\s*%", "sanity_loss", "-"),
    (r"La COR se reduce más fácilmente:?\s*\+?(\d+(?:\.\d+)?)\s*%", "sanity_loss", "+"),
    (r"[Tt]asa de captura\s*([+-]?\s*\d+(?:\.\d+)?)\s*%", "capture_rate", "+"),
)


def _find_element(context: str) -> str | None:
    lowered = context.lower()
    for spanish, code in _ELEMENTS_ES:
        if spanish in lowered:
            return code
    return None


def parse_passive_description(description: str) -> tuple[tuple[str, float, str | None], ...]:
    """A Spanish passive-skill description -> its effects, as (stat, value, element).

    Parses by scanning the whole (whitespace-collapsed) text for every known
    phrase pattern rather than splitting into clauses first: several passives
    describe more than one stat in a single flowing sentence ("Aumenta el
    ataque en un 20%, la defensa en un 20% y la velocidad ..."), and splitting
    on punctuation first only fragments those.

    Phrasing this project has no matching `Stat` for (elemental *resistance*,
    "immune to X", equipment/sphere bonuses) is silently skipped: this
    function reports what it found, not what it failed to find. Coverage
    against a real page is a job for the caller, comparing input and output.
    """
    text = " ".join(description.split())
    effects: list[tuple[str, float, str | None]] = []
    consumed = bytearray(len(text))

    for pattern, stat, sign in _EFFECT_PATTERNS:
        for match in re.finditer(pattern, text):
            raw = match.group(1).replace(" ", "")
            if sign == "+":
                value = float(raw.lstrip("+"))
            elif sign == "-":
                value = -float(raw.lstrip("+"))
            else:
                value = float(raw)
            effects.append((stat, value, None))
            for i in range(match.start(), match.end()):
                consumed[i] = 1

    for match in re.finditer(
        r"[Aa]umenta el daño de los ataques[^%]{0,25}en un\s*(\d+(?:\.\d+)?)\s*%", text
    ):
        element = _find_element(text[max(0, match.start() - 25) : match.end()])
        effects.append(("element_damage", float(match.group(1)), element))
        for i in range(match.start(), match.end()):
            consumed[i] = 1

    return tuple(effects)


_GENERIC_TIER_NAME = re.compile(r"^(.*)\s+Lv\.?\s*\d+$")

# The common shape both sides of the match are reduced to: one passive's
# effects as an order-independent set of (stat, rounded value, element).
EffectSignature = frozenset[tuple[str, float, str | None]]


def effect_signature(effects: list[dict[str, Any]]) -> EffectSignature:
    """A `Passive.effects`-shaped list -> its order-independent signature."""
    return frozenset((e["stat"], round(e["value"], 1), e.get("element")) for e in effects)


def _card_signature(effects: tuple[tuple[str, float, str | None], ...]) -> EffectSignature:
    """A `parse_passive_description`-shaped tuple -> the same kind of signature."""
    return frozenset((stat, round(value, 1), element) for stat, value, element in effects)


def match_passive_translations(
    passives: list[dict[str, Any]], cards: list[dict[str, Any]]
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Match Spanish passive-skill cards to this project's passives by effect.

    `passives` is this project's own passive list (each a dict with at least
    `id`, `name`, `effects`: a list of `{stat, value, element}`). `cards` is a
    list of `{name, effects}` where `effects` is what `parse_passive_description`
    returns, one entry per Spanish card with a non-empty parse.

    Returns the resolved `id -> Spanish name` map, plus the Spanish names that
    could not be resolved uniquely -- reported, not guessed, per this
    project's rule for every other kind of gap.
    """
    by_signature: dict[EffectSignature, list[dict[str, Any]]] = {}
    for passive in passives:
        sig = effect_signature(passive.get("effects", []))
        if sig:
            by_signature.setdefault(sig, []).append(passive)

    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    for card in cards:
        sig = _card_signature(card["effects"])
        if not sig:
            continue
        candidates = by_signature.get(sig, [])
        if len(candidates) == 1:
            resolved[candidates[0]["id"]] = card["name"]
            continue
        specific = [p for p in candidates if not _GENERIC_TIER_NAME.match(p["name"])]
        if len(specific) == 1:
            resolved[specific[0]["id"]] = card["name"]
        elif candidates:
            unresolved.append(card["name"])

    return resolved, tuple(unresolved)


# -- network fetch (optional: needs Playwright) ------------------------------


@dataclass
class PalworldGGSource:
    """Rebuilds the Spanish translation set from the live site.

    Requires `playwright` (`pip install playwright && playwright install
    chromium`), which is not a core dependency of this project: translation
    data changes far less often than the roster does, so most users will
    never need to run this. The generated file is what ships in
    `data/i18n/es.json` and is committed like every other regenerable dataset
    here.

    Unlike `gamefiles.py`, this cannot be a plain HTTP fetch: the passive-skill
    catalog paginates by re-rendering client-side with no separate URL per
    page, so getting all of it means driving a real browser through the
    pagination control. The pal database, by contrast, is a single compiled
    JS module and does not need a browser at all -- `extract_pal_names` works
    on it directly -- but fetching it here anyway keeps this one class the
    single source for regeneration.

    The site's build assigns a fresh hash to every JS filename on each
    deploy, so the pal-database chunk cannot be hardcoded by name: it is
    found by content, not by filename (see `_find_pal_chunk`).
    """

    base_url: str = "https://palworld.gg"
    locale: str = "es"
    min_delay: float = 1.5
    _warnings: list[str] = field(default_factory=list, init=False)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    def fetch(self, dataset_pals: list[dict[str, Any]], dataset_passives: list[dict[str, Any]]) -> dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise IngestError(
                "the i18n source needs Playwright: pip install playwright && "
                "playwright install chromium"
            ) from exc

        known_pal_ids = {p["id"] for p in dataset_pals}
        retrieved_at = datetime.now(UTC).isoformat()

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()

            pal_chunk = self._find_pal_chunk(page)
            all_names = extract_pal_names(pal_chunk)
            pal_names = {pid: name for pid, name in all_names.items() if pid in known_pal_ids}
            missing = known_pal_ids - pal_names.keys()
            if missing:
                self._warnings.append(f"no ES name found for {len(missing)} pal(s): {sorted(missing)[:10]}")

            cards = self._collect_passive_cards(page)
            browser.close()

        parsed_cards = []
        for card in cards:
            if card["source"] not in ("PALS", "ÁRBOL DEL MUNDO", "MUTACIÓN", "LEGENDARIO"):
                continue
            if "es_text" in card["name"].lower():
                continue
            effects = parse_passive_description(card["descr"])
            if effects:
                parsed_cards.append({"name": card["name"], "effects": effects})

        passive_names, unresolved = match_passive_translations(dataset_passives, parsed_cards)
        if unresolved:
            self._warnings.append(
                f"{len(unresolved)} passive card(s) matched more than one candidate and were "
                f"left untranslated: {list(unresolved)}"
            )

        return {
            "locale": self.locale,
            "retrieved_at": retrieved_at,
            "pals": pal_names,
            "passives": passive_names,
        }

    def _find_pal_chunk(self, page: Any) -> str:
        """The compiled pal-database module's source, found by content.

        Every `<script type="module">` on the pals page is a candidate; the
        one that actually is the pal database is recognised by containing a
        pal record's opening shape, not by its (deploy-specific) filename.
        """
        page.goto(f"{self.base_url}/{self.locale}/pals", wait_until="networkidle", timeout=30000)
        scripts = page.eval_on_selector_all(
            "script[type=module][src]", "els => els.map(e => e.src)"
        )
        for src in scripts:
            response = page.request.get(src)
            if not response.ok:
                continue
            body = response.text()
            if _PAL_RECORD_START.search(body):
                return body
        raise IngestError(
            f"no script on {self.base_url}/{self.locale}/pals looks like the pal database; "
            f"the site's structure may have changed"
        )

    def _collect_passive_cards(self, page: Any) -> list[dict[str, str]]:
        """Every passive-skill card, clicking through pagination to get them all."""
        page.goto(
            f"{self.base_url}/{self.locale}/passive-skills", wait_until="networkidle", timeout=30000
        )
        page.wait_for_timeout(500)

        cards: list[dict[str, str]] = []
        seen_pages: set[str] = set()
        for _ in range(20):  # generous cap; real pagination is a handful of pages
            for article in page.query_selector_all("article.passive-skill"):
                name_el = article.query_selector(".name")
                descr_el = article.query_selector(".descr")
                rank_el = article.query_selector(".rank-badge img")
                source_el = article.query_selector(".meta span")
                if not name_el:
                    continue
                cards.append(
                    {
                        "name": name_el.inner_text().strip(),
                        "descr": descr_el.inner_text().strip() if descr_el else "",
                        "rank": rank_el.get_attribute("alt") if rank_el else "",
                        "source": source_el.inner_text().strip() if source_el else "",
                    }
                )

            next_button = page.query_selector("button[aria-label='next'], button:has-text('›')")
            if not next_button or next_button.is_disabled():
                break
            marker = page.content()[:200]
            if marker in seen_pages:
                break
            seen_pages.add(marker)
            next_button.click()
            page.wait_for_timeout(500)

        return cards
