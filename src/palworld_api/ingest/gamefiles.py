"""Ingest from the game's own DataTables.

This is the authoritative source. Everything here comes out of
`DT_PalMonsterParameter`, `DT_PalCombiUnique` and `DT_PassiveSkill_Main`, which
are the tables the game itself reads, so there is no parsing guesswork and no
wiki lag behind a patch.

It expects a directory tree in the shape UE asset extractors produce:

    <root>/Pal/Content/Pal/DataTable/Character/DT_PalMonsterParameter.json
    <root>/Pal/Content/Pal/DataTable/Character/DT_PalCombiUnique.json
    <root>/Pal/Content/Pal/DataTable/PassiveSkill/DT_PassiveSkill_Main.json
    <root>/Pal/Content/L10N/<locale>/Pal/DataTable/Text/DT_PalNameText_Common.json
    <root>/Pal/Content/L10N/<locale>/Pal/DataTable/Text/DT_SkillNameText_Common.json

Two things the tables encode that are easy to get wrong:

- `IgnoreCombi` marks a pal the rank formula must never touch, as parent or as
  child. Those pals still breed, but only through an explicit unique combo --
  which is how the legendaries end up able to produce nothing but themselves.
- The table carries ~750 rows, most of which are boss, summon, raid and arena
  variants of the same creature. Only rows with a real paldeck index and a real
  breeding rank are pals in the sense this API means.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..models import (
    Confidence,
    Dataset,
    Element,
    Gender,
    InheritanceConfig,
    Pal,
    PalBaseStats,
    Passive,
    PassiveEffect,
    Provenance,
    SpecialCombo,
    Stat,
    Work,
)
from .base import IngestError

# Rows the formula must never see: summoned copies, arena and rig variants, and
# the "max level" duplicates. They share a paldeck index with the real pal and
# carry a sentinel breeding rank.
VARIANT_PATTERN = re.compile(
    r"(^SUMMON_)|(_MAX$)|(_Oilrig$)|(_Predator$)|(_BOSS$)|(_Tower\w*$)|(_RAID\w*$)",
    re.IGNORECASE,
)
# The sentinel the table uses for "not breedable by rank".
UNBREEDABLE_RANK = 9999

ELEMENTS: dict[str, Element] = {
    "Normal": Element.NEUTRAL,
    "Fire": Element.FIRE,
    "Water": Element.WATER,
    "Leaf": Element.GRASS,
    "Electricity": Element.ELECTRIC,
    "Ice": Element.ICE,
    "Earth": Element.GROUND,
    "Dark": Element.DARK,
    "Dragon": Element.DRAGON,
}

WORK_FIELDS: dict[str, Work] = {
    "EmitFlame": Work.KINDLING,
    "Watering": Work.WATERING,
    "Seeding": Work.PLANTING,
    "GenerateElectricity": Work.GENERATING_ELECTRICITY,
    "Handcraft": Work.HANDIWORK,
    "Collection": Work.GATHERING,
    "Deforest": Work.LUMBERING,
    "Mining": Work.MINING,
    "ProductMedicine": Work.MEDICINE_PRODUCTION,
    "Cool": Work.COOLING,
    "Transport": Work.TRANSPORTING,
    "MonsterFarm": Work.FARMING,
    "OilExtraction": Work.OIL_EXTRACTION,
}

# Passive effect types that map onto a stat this API scores along. Anything not
# listed is kept on the passive as an unmapped effect rather than dropped.
EFFECTS: dict[str, Stat] = {
    "MaxHP": Stat.HP,
    "ShotAttack": Stat.ATTACK,
    "MeleeAttack": Stat.ATTACK,
    "Defense": Stat.DEFENSE,
    "CraftSpeed": Stat.WORK_SPEED,
    "MoveSpeed": Stat.MOVEMENT_SPEED,
    "Sanity_Decrease": Stat.SANITY_LOSS,
    "FullStomatch_Decrease": Stat.HUNGER_LOSS,
    "CaptureLevel": Stat.CAPTURE_RATE,
}
GENDERS: dict[str, Gender] = {"Male": Gender.MALE, "Female": Gender.FEMALE}

ELEMENT_BOOST_PREFIX = "ElementBoost_"
WORK_RANK_PREFIX = "WorkSuitabilityAddRank_"


def _enum_tail(value: Any) -> str:
    """`EPalElementType::Earth` -> `Earth`."""
    return str(value).rsplit("::", 1)[-1]


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    """Read the `Rows` map out of an extracted UE DataTable."""
    if not path.exists():
        raise IngestError(f"missing DataTable: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IngestError(f"{path} is not valid JSON: {exc}") from exc

    # Extractors wrap the table in a single-element list of export objects.
    entries = payload if isinstance(payload, list) else [payload]
    for entry in entries:
        if isinstance(entry, dict) and isinstance(entry.get("Rows"), dict):
            return entry["Rows"]
    raise IngestError(f"{path} has no Rows map; is it an extracted DataTable?")


def _fold_keys(rows: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Case-folded view of a localisation table.

    The text tables do not always capitalise a row id the same way the monster
    table does -- `WindChimes` is looked up as `PAL_NAME_Windchimes` -- so an
    exact-match lookup silently loses names. Folding once up front avoids that.
    """
    return {key.casefold(): value for key, value in rows.items()}


def _localized(rows: dict[str, dict[str, Any]], key: str) -> str | None:
    """Pull a display string out of a case-folded localisation table."""
    entry = rows.get(key.casefold())
    if not entry:
        return None
    text = entry.get("TextData", {}).get("LocalizedString")
    if not text or text == "en Text":
        # Placeholder rows for unshipped content; treat as absent.
        return None
    return str(text)


def slug(raw: str) -> str:
    """`LazyDragon_Electric` -> `lazydragon_electric`."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").lower()
    if not cleaned:
        raise IngestError(f"cannot build an id from {raw!r}")
    return cleaned


@dataclass
class GameFilesSource:
    """Builds a `Dataset` from extracted game DataTables."""

    root: Path
    name: str = "gamefiles"
    locale: str = "en"
    game_version: str | None = None
    _warnings: list[str] = field(default_factory=list, init=False)

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    # -- paths ------------------------------------------------------------

    @property
    def _content(self) -> Path:
        return Path(self.root) / "Pal" / "Content"

    def _table(self, *parts: str) -> Path:
        return self._content.joinpath(*parts)

    def _text_table(self, name: str) -> Path:
        return self._content / "L10N" / self.locale / "Pal" / "DataTable" / "Text" / name

    # -- entry point ------------------------------------------------------

    def fetch(self) -> Dataset:
        monsters = _rows(self._table("Pal", "DataTable", "Character", "DT_PalMonsterParameter.json"))
        combos_raw = _rows(self._table("Pal", "DataTable", "Character", "DT_PalCombiUnique.json"))
        passives_raw = _rows(self._table("Pal", "DataTable", "PassiveSkill", "DT_PassiveSkill_Main.json"))
        pal_names = _fold_keys(_rows(self._text_table("DT_PalNameText_Common.json")))
        skill_names = _fold_keys(_rows(self._text_table("DT_SkillNameText_Common.json")))

        retrieved_at = datetime.now(UTC).isoformat()
        provenance = Provenance(
            source=self.name,
            confidence=Confidence.GAME_FILES,
            retrieved_at=retrieved_at,
            game_version=self.game_version,
        )

        roster = self._select_pals(monsters, pal_names)
        combos, breed_only = self._build_combos(combos_raw, roster, provenance)
        # Passives are built first so each pal's innate-passive references
        # (PassiveSkill1-4) can be checked against real ids as they're read,
        # rather than validated in a second pass.
        passives = self._build_passives(passives_raw, skill_names, provenance)
        known_passive_ids = {p.id for p in passives}
        pals = tuple(
            self._build_pal(raw_id, row, pal_names, breed_only, known_passive_ids, provenance)
            for raw_id, row in roster.items()
        )

        return Dataset(
            game_version=self.game_version,
            generated_at=retrieved_at,
            source=self.name,
            pals=pals,
            passives=passives,
            special_combos=combos,
            inheritance=InheritanceConfig(
                provenance=Provenance(
                    source="community",
                    confidence=Confidence.UNKNOWN,
                    retrieved_at=retrieved_at,
                )
            ),
        )

    # -- pals -------------------------------------------------------------

    def _select_pals(
        self, monsters: dict[str, dict[str, Any]], pal_names: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Cut ~750 table rows down to the pals a player can actually own.

        A real pal has a paldeck index, a usable breeding rank, and a shipped
        display name. Everything else in the table is a combat variant, a quest
        prop, or content that never shipped -- the name check is what separates
        those last two from the real thing, since they keep a paldeck index
        borrowed from the pal they were cloned off.
        """
        roster: dict[str, dict[str, Any]] = {}
        for raw_id, row in monsters.items():
            if VARIANT_PATTERN.search(raw_id):
                continue
            if int(row.get("ZukanIndex") or 0) <= 0:
                continue
            if int(row.get("CombiRank") or 0) >= UNBREEDABLE_RANK:
                # Kept out of the roster entirely: with no real rank it could
                # never take part in breeding, and listing it would only make
                # routes look reachable when they are not.
                continue
            if _localized(pal_names, f"PAL_NAME_{raw_id}") is None:
                continue
            roster[raw_id] = row
        if not roster:
            raise IngestError(
                "no pals survived filtering; the DataTable layout may have changed"
            )
        return roster

    def _build_pal(
        self,
        raw_id: str,
        row: dict[str, Any],
        pal_names: dict[str, dict[str, Any]],
        breed_only: set[str],
        known_passive_ids: set[str],
        provenance: Provenance,
    ) -> Pal:
        # Guaranteed present: the roster filter requires it.
        name = _localized(pal_names, f"PAL_NAME_{raw_id}") or raw_id

        elements = tuple(
            dict.fromkeys(
                element
                for key in ("ElementType1", "ElementType2")
                if (element := ELEMENTS.get(_enum_tail(row.get(key, "None")))) is not None
            )
        )

        work = {
            suitability: level
            for suffix, suitability in WORK_FIELDS.items()
            if (level := int(row.get(f"WorkSuitability_{suffix}") or 0)) > 0
        }

        stats = {
            stat: float(row[key])
            for key, stat in (
                ("Hp", Stat.HP),
                ("MeleeAttack", Stat.ATTACK),
                ("Defense", Stat.DEFENSE),
                ("CraftSpeed", Stat.WORK_SPEED),
            )
            if row.get(key) is not None
        }
        base_stats = self._build_base_stats(row)
        innate_passives = self._build_innate_passives(raw_id, row, known_passive_ids)

        # IgnoreCombi excludes a pal from the rank formula in both directions.
        # It can still appear in a unique combo, which the engine checks first.
        ignores_combi = bool(row.get("IgnoreCombi"))

        male_probability = row.get("MaleProbability")
        if male_probability is not None and not 0 <= int(male_probability) <= 100:
            # The table uses -1 for "no gender"; do not pass that off as a ratio.
            male_probability = None

        suffix = str(row.get("ZukanIndexSuffix") or "")
        return Pal(
            id=slug(raw_id),
            name=name,
            paldeck=f"{int(row['ZukanIndex']):03d}{suffix}",
            elements=elements,
            combi_rank=int(row["CombiRank"]),
            breed_order=int(row.get("CombiDuplicatePriority") or 0),
            wild_obtainable=slug(raw_id) not in breed_only,
            breedable_as_parent=not ignores_combi,
            breedable_as_child=not ignores_combi,
            male_probability=int(male_probability) if male_probability is not None else None,
            rarity=int(row["Rarity"]) if row.get("Rarity") is not None else None,
            work=work,
            stats=stats,
            base_stats=base_stats,
            innate_passives=innate_passives,
            provenance=provenance,
        )

    @staticmethod
    def _build_base_stats(row: dict[str, Any]) -> PalBaseStats:
        """The full combat/movement stat sheet, one field per named source field."""
        fields = {
            "hp": "Hp",
            "melee_attack": "MeleeAttack",
            "ranged_attack": "ShotAttack",
            "defense": "Defense",
            "support": "Support",
            "stamina": "Stamina",
            "run_speed": "RunSpeed",
            "mount_run_speed": "RideSprintSpeed",
            "walk_speed": "SlowWalkSpeed",
            "price": "Price",
        }
        return PalBaseStats(
            **{
                field: float(row[key])
                for field, key in fields.items()
                if row.get(key) is not None
            }
        )

    def _build_innate_passives(
        self, raw_id: str, row: dict[str, Any], known_passive_ids: set[str]
    ) -> tuple[str, ...]:
        """This pal's guaranteed passives, from its PassiveSkill1-4 fields.

        A reference to a skill id that didn't make it into the final passive
        catalog (test content, an unnamed row) is dropped rather than left to
        fail the dataset's referential-integrity check, and reported so the
        drop is visible instead of silent.
        """
        found: list[str] = []
        for slot in (1, 2, 3, 4):
            raw_skill = row.get(f"PassiveSkill{slot}")
            if not raw_skill or raw_skill == "None":
                continue
            passive_id = slug(raw_skill)
            if passive_id in known_passive_ids:
                found.append(passive_id)
            else:
                self._warnings.append(
                    f"{raw_id}: innate passive {raw_skill!r} is not in the passive "
                    f"catalog (test/unnamed content); dropped"
                )
        return tuple(found)

    # -- combos -----------------------------------------------------------

    def _build_combos(
        self,
        combos_raw: dict[str, dict[str, Any]],
        roster: dict[str, dict[str, Any]],
        provenance: Provenance,
    ) -> tuple[tuple[SpecialCombo, ...], set[str]]:
        """Unique parent pairs, plus the set of pals only obtainable through one.

        A pal that appears solely as the child of a cross-species unique pair is
        not catchable in the world. Self-pairs are excluded from that rule: a
        legendary breeding true is still a pal you caught.
        """
        known = {slug(raw_id) for raw_id in roster}
        combos: list[SpecialCombo] = []
        breed_only: set[str] = set()
        seen: set[tuple[str, str, str]] = set()

        for key, row in combos_raw.items():
            parent_a = slug(_enum_tail(row.get("ParentTribeA", "")))
            parent_b = slug(_enum_tail(row.get("ParentTribeB", "")))
            child = slug(str(row.get("ChildCharacterID", "")))

            missing = {parent_a, parent_b, child} - known
            if missing:
                # Combos referencing rows the roster filter removed (unreleased
                # or variant-only content) cannot be represented honestly.
                self._warnings.append(
                    f"combo {key}: skipped, unknown pals {sorted(missing)}"
                )
                continue

            gender_a = _enum_tail(row.get("ParentGenderA", "None"))
            gender_b = _enum_tail(row.get("ParentGenderB", "None"))
            genders: tuple[Gender, Gender] | None = None
            if gender_a in GENDERS and gender_b in GENDERS:
                genders = (GENDERS[gender_a], GENDERS[gender_b])

            ordered = (parent_a, parent_b)
            if parent_a > parent_b:
                ordered = (parent_b, parent_a)
                if genders is not None:
                    genders = (genders[1], genders[0])

            # Deduplicate on the pair AND the child: a pair that makes two
            # different children depending on the parents' sexes is two combos,
            # not a duplicate, and keeping only the first loses one of them.
            fingerprint = (*ordered, child)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            combos.append(
                SpecialCombo(
                    parents=ordered,
                    child=child,
                    parent_genders=genders,
                    provenance=provenance,
                )
            )
            if parent_a != parent_b:
                breed_only.add(child)

        if not combos:
            raise IngestError("no unique breeding combos parsed; check DT_PalCombiUnique")
        return tuple(combos), breed_only

    # -- passives ---------------------------------------------------------

    def _build_passives(
        self,
        passives_raw: dict[str, dict[str, Any]],
        skill_names: dict[str, dict[str, Any]],
        provenance: Provenance,
    ) -> tuple[Passive, ...]:
        """Every passive that ships with a display name and a usable effect."""
        passives: list[Passive] = []
        for raw_id, row in passives_raw.items():
            name = _localized(skill_names, f"PASSIVE_{raw_id}")
            if name is None:
                # Unnamed rows are test and unshipped content.
                continue

            effects, work_bonus, unmapped = self._read_effects(row)
            if not effects and not work_bonus:
                # Nothing this API can score; listing it would imply otherwise.
                continue

            passives.append(
                Passive(
                    id=slug(raw_id),
                    name=name,
                    tier=max(-3, min(3, int(row.get("Rank") or 0))),
                    effects=effects,
                    work_bonus=work_bonus,
                    lottery_weight=int(row["LotteryWeight"])
                    if row.get("LotteryWeight") is not None
                    else None,
                    unmapped_effects=unmapped,
                    exclusive_group=self._exclusive_group(name),
                    provenance=provenance,
                )
            )
        if not passives:
            raise IngestError("no passives parsed; check DT_PassiveSkill_Main")
        return tuple(passives)

    def _read_effects(
        self, row: dict[str, Any]
    ) -> tuple[tuple[PassiveEffect, ...], dict[Work, int], tuple[str, ...]]:
        effects: list[PassiveEffect] = []
        work_bonus: dict[Work, int] = {}
        unmapped: list[str] = []

        for slot in (1, 2, 3):
            effect_type = _enum_tail(row.get(f"EffectType{slot}", "no"))
            if effect_type in ("no", "None", ""):
                continue
            value = float(row.get(f"EffectValue{slot}") or 0.0)
            if value == 0.0:
                continue

            stat = EFFECTS.get(effect_type)
            if stat is not None:
                effects.append(PassiveEffect(stat=stat, value=value))
                continue

            if effect_type.startswith(ELEMENT_BOOST_PREFIX):
                element = ELEMENTS.get(effect_type[len(ELEMENT_BOOST_PREFIX) :])
                if element is not None:
                    effects.append(
                        PassiveEffect(
                            stat=Stat.ELEMENT_DAMAGE, value=value, element=element
                        )
                    )
                    continue

            if effect_type.startswith(WORK_RANK_PREFIX):
                suitability = WORK_FIELDS.get(effect_type[len(WORK_RANK_PREFIX) :])
                if suitability is not None:
                    work_bonus[suitability] = int(value)
                    continue

            unmapped.append(effect_type)

        return tuple(effects), work_bonus, tuple(dict.fromkeys(unmapped))

    @staticmethod
    def _exclusive_group(name: str) -> str | None:
        """Group the tiered families so the optimiser picks only the best tier.

        The game names them "Attack Up Lv. 3" and so on, so the family is just
        the name with the level stripped.
        """
        match = re.match(r"^(.*?)\s*Lv\.?\s*\d+\s*$", name)
        return slug(match.group(1)) if match else None


def iter_locales(root: Path) -> Iterable[str]:
    """Locales present in an extraction, for error messages and tooling."""
    l10n = Path(root) / "Pal" / "Content" / "L10N"
    if not l10n.is_dir():
        return ()
    return sorted(p.name for p in l10n.iterdir() if p.is_dir())
