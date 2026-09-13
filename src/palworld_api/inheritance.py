"""Passive inheritance: the odds, and what they cost in eggs.

The species route tells you *which* pals to breed. It says nothing about the
part that actually takes the time, which is re-rolling eggs until the child
keeps the passives you want. This module answers that half.

The model, matching how inheritance is understood to work in game:

1. The child's candidate pool is the union of both parents' passives.
2. It inherits exactly N of them, N drawn from `inherited_count_weights`.
3. The N are drawn uniformly without replacement from the pool.

Step 3 is what makes an exact answer possible: the probability that a specific
set of `k` passives all survive a draw of `N` from a pool of `M` is just
``C(M-k, N-k) / C(M, N)``, so the whole thing is a weighted sum of
hypergeometric terms and needs no simulation.

The constants in `InheritanceConfig` are community-measured, not read out of the
game, and they shift between patches. They live in config with their own
provenance for exactly that reason.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from .dataset import PalIndex
from .models import InheritanceConfig
from .routes import RoutePlan


@dataclass(frozen=True, slots=True)
class PassiveOdds:
    """The odds of one breeding step producing the passives you want."""

    desired: tuple[str, ...]
    pool_size: int
    probability: float
    expected_attempts: float | None
    attempts_for_90pct: int | None
    attempts_for_99pct: int | None
    # Desired passives that are not in either parent, so no amount of
    # re-rolling this pair can produce them.
    unavailable: tuple[str, ...] = ()

    @property
    def is_possible(self) -> bool:
        return self.probability > 0.0


@dataclass(frozen=True, slots=True)
class RouteStepOdds:
    parent_a: str
    parent_b: str
    child: str
    generation: int
    odds: PassiveOdds
    carried: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RoutePassivePlan:
    target: str
    desired: tuple[str, ...]
    steps: tuple[RouteStepOdds, ...]
    # Product of the per-step probabilities: the chance of a clean run with no
    # re-rolls at any step. Usually small, and worth showing for that reason.
    clean_run_probability: float
    total_expected_attempts: float | None
    achieved: tuple[str, ...]
    unreachable_passives: tuple[str, ...] = ()


def subset_survival_probability(
    pool_size: int, desired_count: int, config: InheritanceConfig
) -> float:
    """P(a specific set of `desired_count` passives is entirely inherited).

    Exact, by summing the hypergeometric probability over the distribution of
    how many passives the child inherits at all.
    """
    if desired_count == 0:
        return 1.0
    if pool_size <= 0 or desired_count > pool_size:
        return 0.0

    total = 0.0
    for i, weight in enumerate(config.inherited_count_weights):
        # A roll for more passives than the parents have between them cannot
        # invent any: it just hands over the whole pool. Capping here rather
        # than discarding the roll is what keeps a two-passive pair from
        # looking far less likely than it is.
        inherited = min(i + 1, pool_size)
        if inherited < desired_count:
            continue
        # Choose the remaining (inherited - desired_count) slots from the
        # passives we do not care about; divide by all equally likely draws.
        favourable = math.comb(pool_size - desired_count, inherited - desired_count)
        possible = math.comb(pool_size, inherited)
        total += weight * (favourable / possible)
    return total


def _attempts_for_confidence(probability: float, confidence: float) -> int | None:
    """How many eggs until you are `confidence` sure of at least one success."""
    if probability <= 0.0:
        return None
    if probability >= 1.0:
        return 1
    return math.ceil(math.log1p(-confidence) / math.log1p(-probability))


def odds_for_pair(
    desired: Iterable[str],
    parent_a_passives: Iterable[str],
    parent_b_passives: Iterable[str],
    config: InheritanceConfig,
) -> PassiveOdds:
    """Odds that one pairing yields a child carrying all of `desired`."""
    desired_set = dict.fromkeys(desired)  # preserve order, drop duplicates
    pool = dict.fromkeys([*parent_a_passives, *parent_b_passives])

    unavailable = tuple(p for p in desired_set if p not in pool)
    wanted = tuple(p for p in desired_set if p in pool)

    if unavailable:
        # No re-roll can conjure a passive neither parent has, so the honest
        # answer for the requested set is zero rather than a partial credit.
        return PassiveOdds(
            desired=tuple(desired_set),
            pool_size=len(pool),
            probability=0.0,
            expected_attempts=None,
            attempts_for_90pct=None,
            attempts_for_99pct=None,
            unavailable=unavailable,
        )

    probability = subset_survival_probability(len(pool), len(wanted), config)
    return PassiveOdds(
        desired=tuple(desired_set),
        pool_size=len(pool),
        probability=round(probability, 6),
        expected_attempts=round(1.0 / probability, 2) if probability > 0 else None,
        attempts_for_90pct=_attempts_for_confidence(probability, 0.90),
        attempts_for_99pct=_attempts_for_confidence(probability, 0.99),
    )


class InheritanceEngine:
    """Applies the inheritance model along a planned breeding route.

    Give it a `PalIndex` and it will accept display names as well as ids for
    both pals and passives. Without one, callers must pass ids: a name that
    does not resolve would otherwise be silently treated as a passive nobody
    owns, and the route would come back reporting zero chance for no visible
    reason.
    """

    def __init__(self, config: InheritanceConfig, index: PalIndex | None = None) -> None:
        self.config = config
        self.index = index

    def _passive_id(self, ref: str) -> str:
        if self.index is None:
            return ref
        passive = self.index.resolve_passive(ref)
        return passive.id if passive is not None else ref

    def _pal_id(self, ref: str) -> str:
        if self.index is None:
            return ref
        pal = self.index.resolve(ref)
        return pal.id if pal is not None else ref

    def odds(
        self,
        desired: Iterable[str],
        parent_a_passives: Iterable[str],
        parent_b_passives: Iterable[str],
    ) -> PassiveOdds:
        return odds_for_pair(
            desired, parent_a_passives, parent_b_passives, self.config
        )

    def annotate_route(
        self,
        plan: RoutePlan,
        desired: Sequence[str],
        starting_passives: Mapping[str, Sequence[str]] | None = None,
    ) -> RoutePassivePlan:
        """Walk a species route and cost out the passive re-rolls at each step.

        `starting_passives` says which passives your existing pals already carry.
        At each step the plan assumes you re-roll until the child keeps every
        desired passive that its parents can actually supply, which is how the
        breeding is done in practice, and reports what that costs.
        """
        desired_order = tuple(dict.fromkeys(self._passive_id(d) for d in desired))
        desired_set = set(desired_order)

        # Passives each line is carrying, as the route progresses. Both the pal
        # and the passive references are normalised to ids here, because the
        # route's steps are expressed in ids and a mismatch would look exactly
        # like "no parent carries this".
        pools: dict[str, set[str]] = {
            self._pal_id(pal): {self._passive_id(p) for p in passives}
            for pal, passives in (starting_passives or {}).items()
        }
        for pal in plan.owned:
            pools.setdefault(pal, set())

        steps: list[RouteStepOdds] = []
        clean_run = 1.0
        total_attempts = 0.0
        attempts_known = True

        for step in plan.steps:
            pool_a = pools.get(step.parent_a, set())
            pool_b = pools.get(step.parent_b, set())
            available = (pool_a | pool_b) & desired_set

            step_odds = self.odds(sorted(available), pool_a, pool_b)
            # The child carries forward only what we bred it to carry; anything
            # outside `desired` is noise we do not track, since we would re-roll
            # it away rather than rely on it.
            carried = frozenset(available)
            pools[step.child] = set(carried)

            steps.append(
                RouteStepOdds(
                    parent_a=step.parent_a,
                    parent_b=step.parent_b,
                    child=step.child,
                    generation=step.generation,
                    odds=step_odds,
                    carried=tuple(sorted(carried)),
                )
            )

            clean_run *= step_odds.probability
            if step_odds.expected_attempts is None:
                attempts_known = False
            else:
                total_attempts += step_odds.expected_attempts

        achieved = tuple(sorted(pools.get(plan.target, set()) & desired_set))
        return RoutePassivePlan(
            target=plan.target,
            desired=desired_order,
            steps=tuple(steps),
            clean_run_probability=round(clean_run, 8),
            total_expected_attempts=(
                round(total_attempts, 2) if attempts_known else None
            ),
            achieved=achieved,
            unreachable_passives=tuple(
                p for p in desired_order if p not in set(achieved)
            ),
        )
