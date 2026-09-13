"""Shared fixtures.

The dataset here is deliberately synthetic: made-up pals with ranks chosen so
the expected breeding results can be worked out by hand. Testing the engine
against real game data would conflate two different failures — a broken
algorithm and a bad scrape — and only the first one is this suite's business.
"""

from __future__ import annotations

import pytest

from palworld_api.breeding import BreedingEngine
from palworld_api.dataset import PalIndex
from palworld_api.models import (
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
from palworld_api.passives import PassiveEngine
from palworld_api.routes import RoutePlanner

SEED = Provenance(source="test", confidence=Confidence.SEED)


def _pal(pal_id: str, rank: int, **kwargs) -> Pal:
    return Pal(
        id=pal_id,
        name=pal_id.upper(),
        combi_rank=rank,
        paldeck=str(rank).zfill(3),
        provenance=SEED,
        **kwargs,
    )


@pytest.fixture(scope="session")
def dataset() -> Dataset:
    pals = [
        _pal("p10", 10, work={Work.MINING: 4}, elements=(Element.GROUND,)),
        _pal("p20", 20),
        # The goal pal: reachable only by breeding, never caught in the wild.
        _pal("apex", 30, wild_obtainable=False, elements=(Element.DARK,)),
        _pal("p50", 50),
        _pal("p100", 100),
        _pal("p200", 200),
        _pal("p400", 400),
        _pal("p800", 800),
        _pal("p1000", 1000),
        # Same rank, different breed_order: exercises the tie-break.
        _pal("twin_late", 3000, breed_order=2),
        _pal("twin_early", 3000, breed_order=1),
        # Can be a parent but never hatched, like a boss-only pal.
        _pal("uncatchable_child", 60, breedable_as_child=False),
        # No rank at all: must be excluded from breeding rather than guessed at.
        Pal(id="rankless", name="RANKLESS", combi_rank=None, provenance=SEED),
    ]
    passives = [
        Passive(
            id="fierce",
            name="Fierce",
            tier=2,
            effects=(PassiveEffect(stat=Stat.ATTACK, value=20.0),),
            exclusive_group="attack",
            provenance=SEED,
        ),
        Passive(
            id="brave",
            name="Brave",
            tier=1,
            effects=(PassiveEffect(stat=Stat.ATTACK, value=10.0),),
            exclusive_group="attack",
            provenance=SEED,
        ),
        Passive(
            id="artisan",
            name="Artisan",
            tier=2,
            effects=(PassiveEffect(stat=Stat.WORK_SPEED, value=20.0),),
            exclusive_group="work",
            provenance=SEED,
        ),
        Passive(
            id="serene",
            name="Serene",
            tier=1,
            effects=(PassiveEffect(stat=Stat.SANITY_LOSS, value=-15.0),),
            provenance=SEED,
        ),
        Passive(
            id="nocturnal",
            name="Nocturnal",
            tier=2,
            effects=(PassiveEffect(stat=Stat.ATTACK, value=30.0),),
            condition="night",
            provenance=SEED,
        ),
        Passive(
            id="clumsy",
            name="Clumsy",
            tier=-1,
            effects=(PassiveEffect(stat=Stat.ATTACK, value=-20.0),),
            provenance=SEED,
        ),
        Passive(
            id="apex_only",
            name="Apex Only",
            tier=3,
            effects=(PassiveEffect(stat=Stat.ATTACK, value=50.0),),
            restricted_to=("apex",),
            provenance=SEED,
        ),
    ]
    combos = [
        # Overrides the formula, which would otherwise give something near 900.
        SpecialCombo(parents=("p800", "p1000"), child="apex", provenance=SEED),
    ]
    return Dataset(
        game_version="test",
        source="test",
        pals=tuple(pals),
        passives=tuple(passives),
        special_combos=tuple(combos),
        inheritance=InheritanceConfig(provenance=SEED),
    )


@pytest.fixture(scope="session")
def index(dataset: Dataset) -> PalIndex:
    return PalIndex(dataset)


@pytest.fixture(scope="session")
def engine(index: PalIndex) -> BreedingEngine:
    return BreedingEngine(index)


@pytest.fixture(scope="session")
def planner(index: PalIndex, engine: BreedingEngine) -> RoutePlanner:
    return RoutePlanner(index, engine)


@pytest.fixture(scope="session")
def passive_engine(index: PalIndex) -> PassiveEngine:
    return PassiveEngine(index)
