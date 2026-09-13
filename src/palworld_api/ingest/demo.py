"""A synthetic dataset so the API can run before any scrape has happened.

The pals here are invented and named as such. That is a deliberate choice: a
demo built from real pal names with placeholder numbers is indistinguishable
from real data once it is three screens deep in a JSON response, and someone
would eventually plan a real breeding run on numbers that were never measured.
Fictional names make the distinction impossible to miss.

Run the scraper to replace this with real data.
"""

from __future__ import annotations

from datetime import UTC, datetime

from ..models import (
    Confidence,
    Dataset,
    Element,
    InheritanceConfig,
    Pal,
    Passive,
    PassiveEffect,
    Provenance,
    SpecialCombo,
    Stat,
    Work,
)

DEMO = Provenance(source="demo", confidence=Confidence.SEED)

# (id, display name, rank, elements, work, wild)
_PALS: tuple[tuple[str, str, int, tuple[Element, ...], dict[Work, int], bool], ...] = (
    ("demo_wooly", "Demo Wooly", 1470, (Element.NEUTRAL,), {Work.FARMING: 1}, True),
    ("demo_sprout", "Demo Sprout", 1380, (Element.GRASS,), {Work.PLANTING: 1}, True),
    ("demo_ember", "Demo Ember", 1290, (Element.FIRE,), {Work.KINDLING: 1}, True),
    ("demo_pebble", "Demo Pebble", 1180, (Element.GROUND,), {Work.MINING: 1}, True),
    ("demo_drip", "Demo Drip", 1090, (Element.WATER,), {Work.WATERING: 2}, True),
    ("demo_spark", "Demo Spark", 980, (Element.ELECTRIC,), {Work.GENERATING_ELECTRICITY: 1}, True),
    ("demo_frost", "Demo Frost", 870, (Element.ICE,), {Work.COOLING: 2}, True),
    ("demo_timber", "Demo Timber", 760, (Element.GRASS,), {Work.LUMBERING: 3}, True),
    ("demo_tinker", "Demo Tinker", 650, (Element.NEUTRAL,), {Work.HANDIWORK: 3}, True),
    ("demo_hauler", "Demo Hauler", 540, (Element.GROUND,), {Work.TRANSPORTING: 3}, True),
    ("demo_medic", "Demo Medic", 470, (Element.WATER,), {Work.MEDICINE_PRODUCTION: 3}, True),
    ("demo_forge", "Demo Forge", 400, (Element.FIRE,), {Work.KINDLING: 4, Work.MINING: 2}, True),
    ("demo_gale", "Demo Gale", 330, (Element.NEUTRAL,), {Work.GATHERING: 3}, True),
    ("demo_shade", "Demo Shade", 270, (Element.DARK,), {Work.HANDIWORK: 2}, True),
    ("demo_glacier", "Demo Glacier", 210, (Element.ICE,), {Work.COOLING: 4}, True),
    ("demo_volt", "Demo Volt", 160, (Element.ELECTRIC,), {Work.GENERATING_ELECTRICITY: 4}, True),
    ("demo_titan", "Demo Titan", 120, (Element.GROUND,), {Work.MINING: 4}, True),
    ("demo_wyrm", "Demo Wyrm", 90, (Element.DRAGON,), {Work.HANDIWORK: 4}, False),
    ("demo_phoenix", "Demo Phoenix", 60, (Element.FIRE,), {Work.KINDLING: 5}, False),
    ("demo_warden", "Demo Warden", 40, (Element.DARK, Element.GROUND), {Work.HANDIWORK: 5}, False),
    ("demo_sovereign", "Demo Sovereign", 20, (Element.DRAGON, Element.DARK), {Work.HANDIWORK: 5}, False),
)

_PASSIVES: tuple[Passive, ...] = (
    Passive(
        id="demo_ferocious",
        name="Demo Ferocious",
        tier=2,
        effects=(PassiveEffect(stat=Stat.ATTACK, value=20.0),),
        exclusive_group="attack",
        provenance=DEMO,
    ),
    Passive(
        id="demo_bold",
        name="Demo Bold",
        tier=1,
        effects=(PassiveEffect(stat=Stat.ATTACK, value=10.0),),
        exclusive_group="attack",
        provenance=DEMO,
    ),
    Passive(
        id="demo_legend",
        name="Demo Legend",
        tier=3,
        effects=(
            PassiveEffect(stat=Stat.ATTACK, value=20.0),
            PassiveEffect(stat=Stat.DEFENSE, value=20.0),
            PassiveEffect(stat=Stat.MOVEMENT_SPEED, value=15.0),
        ),
        provenance=DEMO,
    ),
    Passive(
        id="demo_stalwart",
        name="Demo Stalwart",
        tier=2,
        effects=(PassiveEffect(stat=Stat.DEFENSE, value=20.0),),
        exclusive_group="defense",
        provenance=DEMO,
    ),
    Passive(
        id="demo_sturdy",
        name="Demo Sturdy",
        tier=1,
        effects=(PassiveEffect(stat=Stat.DEFENSE, value=10.0),),
        exclusive_group="defense",
        provenance=DEMO,
    ),
    Passive(
        id="demo_artisan",
        name="Demo Artisan",
        tier=2,
        effects=(PassiveEffect(stat=Stat.WORK_SPEED, value=50.0),),
        exclusive_group="work_speed",
        provenance=DEMO,
    ),
    Passive(
        id="demo_diligent",
        name="Demo Diligent",
        tier=1,
        effects=(PassiveEffect(stat=Stat.WORK_SPEED, value=25.0),),
        exclusive_group="work_speed",
        provenance=DEMO,
    ),
    Passive(
        id="demo_serene",
        name="Demo Serene",
        tier=2,
        effects=(PassiveEffect(stat=Stat.SANITY_LOSS, value=-15.0),),
        exclusive_group="sanity",
        provenance=DEMO,
    ),
    Passive(
        id="demo_calm",
        name="Demo Calm",
        tier=1,
        effects=(PassiveEffect(stat=Stat.SANITY_LOSS, value=-10.0),),
        exclusive_group="sanity",
        provenance=DEMO,
    ),
    Passive(
        id="demo_light_eater",
        name="Demo Light Eater",
        tier=2,
        effects=(PassiveEffect(stat=Stat.HUNGER_LOSS, value=-15.0),),
        provenance=DEMO,
    ),
    Passive(
        id="demo_swift",
        name="Demo Swift",
        tier=1,
        effects=(PassiveEffect(stat=Stat.MOVEMENT_SPEED, value=30.0),),
        provenance=DEMO,
    ),
    Passive(
        id="demo_nightstalker",
        name="Demo Nightstalker",
        tier=2,
        effects=(PassiveEffect(stat=Stat.ATTACK, value=30.0),),
        condition="night",
        provenance=DEMO,
    ),
    Passive(
        id="demo_dragonheart",
        name="Demo Dragonheart",
        tier=2,
        effects=(
            PassiveEffect(
                stat=Stat.ELEMENT_DAMAGE, value=20.0, element=Element.DRAGON
            ),
        ),
        provenance=DEMO,
    ),
    Passive(
        id="demo_burdened",
        name="Demo Burdened",
        tier=-1,
        effects=(PassiveEffect(stat=Stat.WORK_SPEED, value=-30.0),),
        provenance=DEMO,
    ),
    Passive(
        id="demo_brittle",
        name="Demo Brittle",
        tier=-2,
        effects=(PassiveEffect(stat=Stat.DEFENSE, value=-20.0),),
        provenance=DEMO,
    ),
    Passive(
        id="demo_crown",
        name="Demo Crown",
        tier=3,
        effects=(PassiveEffect(stat=Stat.ATTACK, value=40.0),),
        restricted_to=("demo_sovereign",),
        heritable=False,
        provenance=DEMO,
    ),
)

_COMBOS: tuple[SpecialCombo, ...] = (
    SpecialCombo(parents=("demo_wyrm", "demo_phoenix"), child="demo_sovereign", provenance=DEMO),
    SpecialCombo(parents=("demo_titan", "demo_shade"), child="demo_warden", provenance=DEMO),
)


def build_demo_dataset() -> Dataset:
    pals = tuple(
        Pal(
            id=pal_id,
            name=name,
            paldeck=str(index + 1).zfill(3),
            elements=elements,
            combi_rank=rank,
            work=work,
            wild_obtainable=wild,
            stats={
                Stat.ATTACK: float(200 - rank // 10),
                Stat.DEFENSE: float(180 - rank // 12),
                Stat.HP: float(400 - rank // 6),
            },
            provenance=DEMO,
        )
        for index, (pal_id, name, rank, elements, work, wild) in enumerate(_PALS)
    )
    return Dataset(
        game_version="demo",
        generated_at=datetime.now(UTC).isoformat(),
        source="demo",
        pals=pals,
        passives=_PASSIVES,
        special_combos=_COMBOS,
        inheritance=InheritanceConfig(provenance=DEMO),
    )
