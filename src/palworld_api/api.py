"""The HTTP API.

Two things this layer takes seriously:

- Every response that depends on game numbers carries the dataset's provenance,
  so a client can tell real data from placeholder data without reading the docs.
- Engine errors map to 4xx with the reason intact. "Not reachable" and "unknown
  pal" are different answers and the caller should not have to guess which it got.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .breeding import BreedingEngine, BreedingError
from .dataset import DatasetError, PalIndex, find_default_dataset, load_dataset
from .ingest.icons import existing_icon_ids
from .inheritance import InheritanceEngine
from .models import Element, Work
from .passives import PROFILES, PassiveEngine, WorkRanking
from .routes import RouteError, RoutePlanner, Strategy
from .translations import Translations, load_translations

_ENV_DATASET = os.environ.get("PALWORLD_DATASET")
DEFAULT_DATASET = Path(_ENV_DATASET) if _ENV_DATASET else find_default_dataset()
WEB_DIR = Path(__file__).parent / "web"
ICONS_DIR = WEB_DIR / "icons"

Lang = Literal["en", "es"]


class Services:
    """Everything the request handlers need, built once at startup."""

    def __init__(self, dataset_path: Path) -> None:
        self.dataset_path = dataset_path
        self.dataset = load_dataset(dataset_path)
        self.index = PalIndex(self.dataset)
        self.breeding = BreedingEngine(self.index)
        self.planner = RoutePlanner(self.index, self.breeding)
        self.passives = PassiveEngine(self.index)
        self.inheritance = InheritanceEngine(self.dataset.inheritance, self.index)
        # Read once at startup rather than stat-ing a file on every /pals
        # request: the icon set only changes when someone re-runs `cli icons`,
        # which restarts the process anyway.
        self.icon_ids = existing_icon_ids(ICONS_DIR)
        # Same reasoning for translations: {} per locale when the sidecar file
        # is absent, so a request for an untranslated locale just falls back
        # to English rather than failing.
        self.translations: dict[str, Translations] = {}
        for locale in ("es",):
            translation = load_translations(find_default_dataset(f"i18n/{locale}.json"))
            if translation is not None:
                self.translations[locale] = translation

    def pal_name(self, pal_id: str, lang: Lang) -> str:
        fallback = self.index.require(pal_id).name
        translation = self.translations.get(lang)
        return translation.pal_name(pal_id, fallback) if translation else fallback

    def passive_name(self, passive_id: str, lang: Lang) -> str:
        fallback = self.index.passives[passive_id].name
        translation = self.translations.get(lang)
        return translation.passive_name(passive_id, fallback) if translation else fallback

    def icon_url(self, pal_id: str) -> str | None:
        """The pal's icon URL, or None when this dataset has no icon for it.

        None rather than a guessed path: a client should never have to load an
        image to find out it doesn't exist.
        """
        return f"/ui/icons/{pal_id}.png" if pal_id in self.icon_ids else None

    @property
    def data_warning(self) -> str | None:
        """A single sentence a client can show verbatim, or None when clean."""
        if self.dataset.source == "demo":
            return (
                "This is the demo dataset. The pals in it are invented and the "
                "numbers are placeholders. Run the ingest CLI for real data."
            )
        if not self.dataset.is_verified:
            count = len(self.index.unverified_pals)
            return f"{count} pal(s) carry unverified values; treat results as provisional."
        return None


@lru_cache(maxsize=1)
def get_services() -> Services:
    try:
        return Services(DEFAULT_DATASET)
    except DatasetError as exc:
        raise RuntimeError(str(exc)) from exc


ServicesDep = Annotated[Services, Depends(get_services)]


class DataQuality(BaseModel):
    source: str
    game_version: str | None
    generated_at: str | None
    verified: bool
    warning: str | None


class PalSummary(BaseModel):
    id: str
    name: str
    paldeck: str | None
    elements: list[Element]
    combi_rank: int | None
    breedable: bool
    wild_obtainable: bool
    icon: str | None = None


class BreedResponse(BaseModel):
    parent_a: str
    parent_b: str
    child: str
    rule: str
    target_rank: int | None
    # Populated only for the rare pairs whose child depends on the parents' sexes.
    alternatives: list[str] = Field(default_factory=list)
    gender_requirement: str | None = None
    data_quality: DataQuality


class RouteStepResponse(BaseModel):
    generation: int
    parent_a: str
    parent_b: str
    child: str
    rule: str
    gender_requirement: str | None = None


class RouteResponse(BaseModel):
    target: str
    strategy: str
    generations: int
    distinct_steps: int
    already_owned: bool
    steps: list[RouteStepResponse]
    intermediates: list[str]
    alternatives_for_final_step: list[list[str]]
    data_quality: DataQuality


class PassiveResponse(BaseModel):
    id: str
    name: str
    score: float
    unconditional_score: float
    condition: str | None
    exclusive_group: str | None


class PassiveSetResponse(BaseModel):
    pal: str
    goal: str
    total_score: float
    considered: int
    passives: list[PassiveResponse]
    data_quality: DataQuality


class WorkRankingEntry(BaseModel):
    pal: str
    name: str
    level: int


class WorkBestResponse(BaseModel):
    work: Work
    pals: list[WorkRankingEntry]
    data_quality: DataQuality


class WorkOverviewResponse(BaseModel):
    by_work: dict[str, list[WorkRankingEntry]]
    data_quality: DataQuality


class InheritanceStepResponse(BaseModel):
    generation: int
    parent_a: str
    parent_b: str
    child: str
    carried: list[str]
    probability: float
    expected_attempts: float | None
    attempts_for_90pct: int | None
    unavailable: list[str]


class InheritanceRequest(BaseModel):
    target: str
    desired_passives: list[str] = Field(default_factory=list)
    owned: list[str] | None = None
    starting_passives: dict[str, list[str]] = Field(default_factory=dict)
    strategy: Strategy = Strategy.GENERATIONS


class InheritanceResponse(BaseModel):
    target: str
    desired: list[str]
    steps: list[InheritanceStepResponse]
    clean_run_probability: float
    total_expected_attempts: float | None
    achieved: list[str]
    unreachable_passives: list[str]
    data_quality: DataQuality


def _gender_note(services: Services, step) -> str | None:
    """The sex requirement for a step, when the pair has more than one child.

    Without this a route can look impossible to follow: the player breeds the
    pair, gets the other child, and has no way to know why.
    """
    pair = (step.parent_a, step.parent_b)
    if pair[0] > pair[1]:
        pair = (pair[1], pair[0])
    combos = services.index.combos_by_pair.get(pair, ())
    if len(combos) < 2:
        return None
    match = next((c for c in combos if c.child == step.child), None)
    return match.describe_genders() if match else None


def _quality(services: Services) -> DataQuality:
    return DataQuality(
        source=services.dataset.source,
        game_version=services.dataset.game_version,
        generated_at=services.dataset.generated_at,
        verified=services.dataset.is_verified,
        warning=services.data_warning,
    )


app = FastAPI(
    title="Palworld Data API",
    version="0.1.0",
    description=(
        "Pal data, breeding combinations, breeding routes and passive "
        "optimisation. Every response carries the provenance of the data "
        "behind it: check `data_quality.verified` before planning on a result."
    ),
)


@app.get("/health", tags=["meta"])
def health(services: ServicesDep) -> dict[str, Any]:
    return {
        "status": "ok",
        "dataset": str(services.dataset_path),
        "pals": len(services.dataset.pals),
        "passives": len(services.dataset.passives),
        "special_combos": len(services.dataset.special_combos),
        "data_quality": _quality(services).model_dump(),
    }


@app.get("/pals", response_model=list[PalSummary], tags=["pals"])
def list_pals(
    services: ServicesDep,
    element: Element | None = None,
    work: Work | None = None,
    breedable: bool | None = None,
    search: str | None = Query(
        default=None, description="Case-insensitive match against the name in any language."
    ),
    lang: Lang = "en",
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> list[PalSummary]:
    pals = list(services.dataset.pals)
    if element is not None:
        pals = [p for p in pals if element in p.elements]
    if work is not None:
        pals = [p for p in pals if p.work.get(work, 0) > 0]
    if breedable is not None:
        pals = [p for p in pals if p.is_breedable_child == breedable]
    if search:
        needle = search.casefold()
        pals = [
            p
            for p in pals
            if needle in p.name.casefold()
            or needle in services.pal_name(p.id, lang).casefold()
        ]

    window = pals[offset : offset + limit]
    return [
        PalSummary(
            id=p.id,
            name=services.pal_name(p.id, lang),
            paldeck=p.paldeck,
            elements=list(p.elements),
            combi_rank=p.combi_rank,
            breedable=p.is_breedable_child,
            wild_obtainable=p.wild_obtainable,
            icon=services.icon_url(p.id),
        )
        for p in window
    ]


@app.get("/pals/{pal_id}", tags=["pals"])
def get_pal(pal_id: str, services: ServicesDep, lang: Lang = "en") -> dict[str, Any]:
    pal = services.index.resolve(pal_id)
    if pal is None:
        raise HTTPException(status_code=404, detail=f"unknown pal: {pal_id!r}")
    body = pal.model_dump(mode="json", exclude_none=True)
    body["name"] = services.pal_name(pal.id, lang)
    return {
        "pal": body,
        "icon": services.icon_url(pal.id),
        "data_quality": _quality(services).model_dump(),
    }


@app.get("/breed", response_model=BreedResponse, tags=["breeding"])
def breed(parent_a: str, parent_b: str, services: ServicesDep) -> BreedResponse:
    """What one pair produces."""
    pal_a = services.index.resolve(parent_a)
    pal_b = services.index.resolve(parent_b)
    if pal_a is None or pal_b is None:
        missing = parent_a if pal_a is None else parent_b
        raise HTTPException(status_code=404, detail=f"unknown pal: {missing!r}")
    try:
        result = services.breeding.breed(pal_a.id, pal_b.id)
    except BreedingError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BreedResponse(
        parent_a=result.parent_a,
        parent_b=result.parent_b,
        child=result.child,
        rule=result.rule,
        target_rank=result.target_rank,
        alternatives=list(result.alternatives),
        gender_requirement=result.gender_requirement,
        data_quality=_quality(services),
    )


@app.get("/pals/{pal_id}/parents", tags=["breeding"])
def parents_of(pal_id: str, services: ServicesDep) -> dict[str, Any]:
    """Every pair that produces this pal."""
    pal = services.index.resolve(pal_id)
    if pal is None:
        raise HTTPException(status_code=404, detail=f"unknown pal: {pal_id!r}")
    pairs = services.breeding.pairs_producing(pal.id)
    return {
        "pal": pal.id,
        "pairs": [list(pair) for pair in pairs],
        "count": len(pairs),
        "data_quality": _quality(services).model_dump(),
    }


@app.get("/pals/{pal_id}/children", tags=["breeding"])
def children_of(pal_id: str, services: ServicesDep) -> dict[str, Any]:
    """What this pal produces with every possible partner."""
    pal = services.index.resolve(pal_id)
    if pal is None:
        raise HTTPException(status_code=404, detail=f"unknown pal: {pal_id!r}")
    children = services.breeding.children_of(pal.id)
    return {
        "pal": pal.id,
        "children": children,
        "count": len(children),
        "data_quality": _quality(services).model_dump(),
    }


@app.get("/routes/{target}", response_model=RouteResponse, tags=["routes"])
def route(
    target: str,
    services: ServicesDep,
    owned: Annotated[
        list[str] | None,
        Query(description="Pals you already have. Defaults to everything catchable."),
    ] = None,
    strategy: Strategy = Strategy.GENERATIONS,
    max_generations: int | None = Query(default=None, ge=1, le=20),
    include_alternatives: bool = True,
) -> RouteResponse:
    """The cheapest breeding route to `target`."""
    try:
        plan = services.planner.plan(
            target, owned=owned, strategy=strategy, max_generations=max_generations
        )
    except RouteError as exc:
        message = str(exc)
        status = 404 if "unknown pal" in message else 422
        raise HTTPException(status_code=status, detail=message) from exc

    alternatives: list[list[str]] = []
    if include_alternatives and not plan.already_owned:
        alternatives = [
            list(pair) for pair in services.planner.alternatives(plan.target, owned)
        ]

    return RouteResponse(
        target=plan.target,
        strategy=plan.strategy.value,
        generations=plan.generations,
        distinct_steps=plan.distinct_steps,
        already_owned=plan.already_owned,
        steps=[
            RouteStepResponse(
                generation=s.generation,
                parent_a=s.parent_a,
                parent_b=s.parent_b,
                child=s.child,
                rule=s.rule,
                gender_requirement=_gender_note(services, s),
            )
            for s in plan.steps
        ],
        intermediates=list(plan.intermediates),
        alternatives_for_final_step=alternatives,
        data_quality=_quality(services),
    )


@app.get("/reachable", tags=["routes"])
def reachable(
    services: ServicesDep,
    owned: Annotated[list[str] | None, Query()] = None,
    max_generations: int | None = Query(default=None, ge=1, le=20),
) -> dict[str, Any]:
    """Every pal reachable from a starting set, with its generation count."""
    try:
        found = services.planner.reachable_from(owned, max_generations=max_generations)
    except RouteError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
        "count": len(found),
        "reachable": found,
        "data_quality": _quality(services).model_dump(),
    }


@app.get("/goals", tags=["passives"])
def goals() -> dict[str, Any]:
    """The built-in passive goals and the weights behind them."""
    return {
        name: {
            "max_slots": profile.max_slots,
            "condition_reliability": profile.condition_reliability,
            "weights": {stat.value: weight for stat, weight in profile.weights.items()},
        }
        for name, profile in PROFILES.items()
    }


@app.get(
    "/pals/{pal_id}/passives", response_model=PassiveSetResponse, tags=["passives"]
)
def best_passives(
    pal_id: str,
    services: ServicesDep,
    goal: str = Query(default="combat", description="One of the ids from /goals."),
    lang: Lang = "en",
) -> PassiveSetResponse:
    """The optimal passive loadout for a pal, for a given goal."""
    try:
        recommendation = services.passives.recommend(pal_id, goal)
    except KeyError as exc:
        message = str(exc).strip("\"'")
        status = 404 if "unknown pal" in message else 422
        raise HTTPException(status_code=status, detail=message) from exc
    return PassiveSetResponse(
        pal=recommendation.pal_id,
        goal=recommendation.goal,
        total_score=recommendation.total_score,
        considered=recommendation.considered,
        passives=[
            PassiveResponse(
                id=p.passive_id,
                name=services.passive_name(p.passive_id, lang),
                score=p.score,
                unconditional_score=p.unconditional_score,
                condition=p.condition,
                exclusive_group=p.exclusive_group,
            )
            for p in recommendation.passives
        ],
        data_quality=_quality(services),
    )


def _entries(ranking: tuple[WorkRanking, ...]) -> list[WorkRankingEntry]:
    return [WorkRankingEntry(pal=r.pal_id, name=r.name, level=r.level) for r in ranking]


@app.get("/work", response_model=WorkOverviewResponse, tags=["passives"])
def best_pals_by_work(
    services: ServicesDep,
    limit: int = Query(default=5, ge=1, le=50, description="Pals per activity."),
) -> WorkOverviewResponse:
    """The best pals for every base activity: watering, planting, mining, and
    so on. This is base work suitability -- the level shown on the pal's own
    card -- not a passive-adjusted score; see `/pals/{id}/passives?goal=base`
    for the passive side of base optimisation instead.
    """
    overview = services.passives.rank_all_work(limit)
    return WorkOverviewResponse(
        by_work={work.value: _entries(ranking) for work, ranking in overview.items()},
        data_quality=_quality(services),
    )


@app.get("/work/{work}/best", response_model=WorkBestResponse, tags=["passives"])
def best_pals_for_work(
    work: Work, services: ServicesDep, limit: int = Query(default=10, ge=1, le=100)
) -> WorkBestResponse:
    """The best pals for one specific activity, most suited first."""
    ranking = services.passives.rank_pals_for_work(work, limit)
    return WorkBestResponse(
        work=work, pals=_entries(ranking), data_quality=_quality(services)
    )


@app.post("/routes/plan", response_model=InheritanceResponse, tags=["routes"])
def plan_with_passives(
    request: InheritanceRequest, services: ServicesDep
) -> InheritanceResponse:
    """A species route costed out for the passives you want to carry through it."""
    try:
        plan = services.planner.plan(
            request.target, owned=request.owned, strategy=request.strategy
        )
    except RouteError as exc:
        message = str(exc)
        status = 404 if "unknown pal" in message else 422
        raise HTTPException(status_code=status, detail=message) from exc

    for passive_id in request.desired_passives:
        if services.index.resolve_passive(passive_id) is None:
            raise HTTPException(
                status_code=404, detail=f"unknown passive: {passive_id!r}"
            )

    annotated = services.inheritance.annotate_route(
        plan,
        desired=request.desired_passives,
        starting_passives=request.starting_passives,
    )
    return InheritanceResponse(
        target=annotated.target,
        desired=list(annotated.desired),
        steps=[
            InheritanceStepResponse(
                generation=s.generation,
                parent_a=s.parent_a,
                parent_b=s.parent_b,
                child=s.child,
                carried=list(s.carried),
                probability=s.odds.probability,
                expected_attempts=s.odds.expected_attempts,
                attempts_for_90pct=s.odds.attempts_for_90pct,
                unavailable=list(s.odds.unavailable),
            )
            for s in annotated.steps
        ],
        clean_run_probability=annotated.clean_run_probability,
        total_expected_attempts=annotated.total_expected_attempts,
        achieved=list(annotated.achieved),
        unreachable_passives=list(annotated.unreachable_passives),
        data_quality=_quality(services),
    )


if WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=WEB_DIR, html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def index_page() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")
