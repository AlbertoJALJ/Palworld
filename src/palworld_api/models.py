"""Canonical data model for the Palworld dataset.

Everything downstream (breeding, routing, passives) reads these types and
nothing else.  The ingest layer's only job is to produce a `Dataset` that
validates against this module, which is what lets the scraper be replaced
without touching the engine.

Provenance is part of the model on purpose: a value that came from a
community wiki and a value read out of the game files are not equally
trustworthy, and the API surfaces that difference instead of hiding it.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator

PalId = Annotated[str, Field(pattern=r"^[a-z0-9_]+$", min_length=1, max_length=64)]
PassiveId = Annotated[str, Field(pattern=r"^[a-z0-9_]+$", min_length=1, max_length=64)]


class Element(StrEnum):
    NEUTRAL = "neutral"
    FIRE = "fire"
    WATER = "water"
    GRASS = "grass"
    ELECTRIC = "electric"
    ICE = "ice"
    GROUND = "ground"
    DARK = "dark"
    DRAGON = "dragon"


class Work(StrEnum):
    """Work suitabilities. Drives the base-usefulness side of passive scoring."""

    KINDLING = "kindling"
    WATERING = "watering"
    PLANTING = "planting"
    GENERATING_ELECTRICITY = "generating_electricity"
    HANDIWORK = "handiwork"
    GATHERING = "gathering"
    LUMBERING = "lumbering"
    MINING = "mining"
    MEDICINE_PRODUCTION = "medicine_production"
    COOLING = "cooling"
    TRANSPORTING = "transporting"
    FARMING = "farming"
    OIL_EXTRACTION = "oil_extraction"


class Stat(StrEnum):
    """Modifiable stats a passive can touch.

    These are the axes the passive optimiser scores along, so the set is
    deliberately small and orthogonal rather than mirroring every in-game
    string.
    """

    ATTACK = "attack"
    DEFENSE = "defense"
    HP = "hp"
    WORK_SPEED = "work_speed"
    MOVEMENT_SPEED = "movement_speed"
    SANITY_LOSS = "sanity_loss"
    HUNGER_LOSS = "hunger_loss"
    CAPTURE_RATE = "capture_rate"
    ELEMENT_DAMAGE = "element_damage"


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"


class Confidence(StrEnum):
    """How much the value should be trusted.

    GAME_FILES wins over WIKI, WIKI wins over SEED. The API exposes this so a
    consumer can refuse to plan on unverified numbers.
    """

    GAME_FILES = "game_files"
    WIKI = "wiki"
    SEED = "seed"
    UNKNOWN = "unknown"


class Provenance(BaseModel):
    """Where a record came from, so a wrong number can be traced and re-pulled."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="Adapter id, e.g. 'paldb' or 'seed'.")
    confidence: Confidence = Confidence.UNKNOWN
    url: str | None = None
    retrieved_at: str | None = Field(
        default=None, description="ISO-8601 UTC timestamp of the fetch."
    )
    game_version: str | None = None

    @property
    def verified(self) -> bool:
        return self.confidence in (Confidence.GAME_FILES, Confidence.WIKI)


class PassiveEffect(BaseModel):
    """A single additive modifier, expressed as a percentage delta.

    `value` is a percentage point change (+10 means +10%). Negative values are
    penalties, which matters because several strong passives carry one.
    """

    model_config = ConfigDict(frozen=True)

    stat: Stat
    value: float
    element: Element | None = Field(
        default=None,
        description="Only set when stat is ELEMENT_DAMAGE and the buff is element-scoped.",
    )

    @model_validator(mode="after")
    def _element_only_for_element_damage(self) -> PassiveEffect:
        if self.element is not None and self.stat is not Stat.ELEMENT_DAMAGE:
            raise ValueError("element may only be set when stat is element_damage")
        return self


class Passive(BaseModel):
    """A passive skill.

    `exclusive_group` models the families where equipping two members is either
    impossible or strictly wasteful; the optimiser will pick at most one member
    of a group. `condition` marks passives that only apply situationally, which
    the optimiser discounts rather than treating as free.
    """

    model_config = ConfigDict(frozen=True)

    id: PassiveId
    name: str
    tier: int = Field(
        default=0,
        ge=-3,
        le=3,
        description="In-game rank, negative for detrimental passives.",
    )
    effects: tuple[PassiveEffect, ...] = ()
    exclusive_group: str | None = None
    condition: str | None = Field(
        default=None, description="Free text, e.g. 'night' or 'hp_below_50'."
    )
    work_bonus: dict[Work, int] = Field(
        default_factory=dict,
        description="Work suitability ranks added outright, not as a percentage.",
    )
    lottery_weight: int | None = Field(
        default=None,
        description="Relative chance of appearing as a random passive. 0 means it "
        "never rolls naturally.",
    )
    unmapped_effects: tuple[str, ...] = Field(
        default=(),
        description="Effects the source declared that this schema has no stat for. "
        "Recorded rather than dropped, so a passive never looks weaker than it is.",
    )
    restricted_to: tuple[PalId, ...] = Field(
        default=(),
        description="If non-empty, only these pals can carry it (innate passives).",
    )
    heritable: bool = Field(
        default=True, description="False for passives that never pass to offspring."
    )
    provenance: Provenance | None = None

    def applies_to(self, pal_id: str) -> bool:
        return not self.restricted_to or pal_id in self.restricted_to


class Pal(BaseModel):
    """A single pal.

    `combi_rank` is the hidden breeding rank that drives the whole breeding
    engine; it is the one field the route finder cannot work without.
    """

    model_config = ConfigDict(frozen=True)

    id: PalId
    name: str
    paldeck: str | None = Field(
        default=None, description="Paldeck number as shown in game, e.g. '099' or '045B'."
    )
    elements: tuple[Element, ...] = ()

    combi_rank: int | None = Field(
        default=None,
        ge=0,
        description="Breeding rank. None means the pal has no known rank and is "
        "excluded from breeding maths rather than guessed at.",
    )
    breed_order: int | None = Field(
        default=None,
        ge=0,
        description="Deterministic tie-break key when two pals sit equally close "
        "to a computed rank. Falls back to paldeck ordering.",
    )

    wild_obtainable: bool = Field(
        default=True,
        description="Catchable in the world without breeding. Seeds the default "
        "starting set for route planning.",
    )
    breedable_as_parent: bool = True
    breedable_as_child: bool = Field(
        default=True,
        description="False for pals that can never be hatched from a generic pair "
        "(uniques reachable only via a special combo, bosses, humans).",
    )

    male_probability: int | None = Field(
        default=None,
        ge=0,
        le=100,
        description="Percent chance of hatching male. A heavily skewed value makes "
        "a pair harder to assemble, even when the route is short.",
    )
    rarity: int | None = Field(default=None, ge=0)
    work: dict[Work, int] = Field(
        default_factory=dict, description="Work suitability levels."
    )
    stats: dict[Stat, float] = Field(
        default_factory=dict, description="Base stat values where known."
    )
    innate_passives: tuple[PassiveId, ...] = ()

    provenance: Provenance | None = None

    @property
    def is_breedable_child(self) -> bool:
        return self.breedable_as_child and self.combi_rank is not None

    @property
    def is_breedable_parent(self) -> bool:
        return self.breedable_as_parent and self.combi_rank is not None


class SpecialCombo(BaseModel):
    """A parent pair that overrides the rank formula.

    Stored unordered: the engine normalises the pair before lookup so
    (a, b) and (b, a) are the same key.

    `parent_genders` exists because a handful of pairs produce a different
    child depending on which parent is which sex. Dropping that distinction
    would silently lose one of the two children the pair can make.
    """

    model_config = ConfigDict(frozen=True)

    parents: tuple[PalId, PalId]
    child: PalId
    parent_genders: tuple[Gender, Gender] | None = Field(
        default=None,
        description="Aligned with `parents`. None means the pair works either way.",
    )
    provenance: Provenance | None = None

    @property
    def key(self) -> tuple[str, str]:
        return tuple(sorted(self.parents))  # type: ignore[return-value]

    @property
    def is_gendered(self) -> bool:
        return self.parent_genders is not None

    def describe_genders(self) -> str | None:
        """Human-readable requirement, e.g. `katress must be female`."""
        if self.parent_genders is None:
            return None
        return ", ".join(
            f"{pal} must be {gender.value}"
            for pal, gender in zip(self.parents, self.parent_genders, strict=True)
        )


class InheritanceConfig(BaseModel):
    """Tunable constants for passive inheritance.

    These are community-measured rather than read out of the game, and they
    move between patches, so they live in config with their own provenance
    instead of being baked into the algorithm.
    """

    model_config = ConfigDict(frozen=True)

    # P(child inherits exactly N passives from the combined parent pool),
    # index 0 -> 1 passive. Must sum to 1.
    inherited_count_weights: tuple[float, ...] = (0.4, 0.3, 0.2, 0.1)
    # Chance each remaining empty slot is filled by a random non-parent passive.
    random_slot_chance: float = Field(default=0.0, ge=0.0, le=1.0)
    max_passive_slots: int = Field(default=4, ge=1, le=8)
    provenance: Provenance | None = None

    @model_validator(mode="after")
    def _weights_sum_to_one(self) -> InheritanceConfig:
        if not self.inherited_count_weights:
            raise ValueError("inherited_count_weights must not be empty")
        total = sum(self.inherited_count_weights)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"inherited_count_weights must sum to 1, got {total}")
        if any(w < 0 for w in self.inherited_count_weights):
            raise ValueError("inherited_count_weights must be non-negative")
        if len(self.inherited_count_weights) > self.max_passive_slots:
            raise ValueError("more inherited-count weights than passive slots")
        return self


class Dataset(BaseModel):
    """The whole world, as one validated document."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = 1
    game_version: str | None = None
    generated_at: str | None = None
    source: str = "unknown"

    pals: tuple[Pal, ...] = ()
    passives: tuple[Passive, ...] = ()
    special_combos: tuple[SpecialCombo, ...] = ()
    inheritance: InheritanceConfig = InheritanceConfig()

    @model_validator(mode="after")
    def _check_referential_integrity(self) -> Dataset:
        pal_ids = {p.id for p in self.pals}
        if len(pal_ids) != len(self.pals):
            raise ValueError("duplicate pal ids in dataset")

        passive_ids = {p.id for p in self.passives}
        if len(passive_ids) != len(self.passives):
            raise ValueError("duplicate passive ids in dataset")

        for combo in self.special_combos:
            unknown = set(combo.parents) | {combo.child}
            missing = unknown - pal_ids
            if missing:
                raise ValueError(
                    f"special combo references unknown pals: {sorted(missing)}"
                )

        for pal in self.pals:
            missing = set(pal.innate_passives) - passive_ids
            if missing:
                raise ValueError(
                    f"pal {pal.id!r} references unknown passives: {sorted(missing)}"
                )

        for passive in self.passives:
            missing = set(passive.restricted_to) - pal_ids
            if missing:
                raise ValueError(
                    f"passive {passive.id!r} restricted to unknown pals: {sorted(missing)}"
                )
        return self

    @property
    def is_verified(self) -> bool:
        """True only when every pal carries a trustworthy provenance."""
        return bool(self.pals) and all(
            p.provenance is not None and p.provenance.verified for p in self.pals
        )
