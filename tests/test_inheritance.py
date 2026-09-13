"""Passive inheritance probabilities and their cost in eggs."""

from __future__ import annotations

import random
from itertools import combinations

import pytest

from palworld_api.inheritance import (
    InheritanceEngine,
    odds_for_pair,
    subset_survival_probability,
)
from palworld_api.models import InheritanceConfig

CONFIG = InheritanceConfig()


def test_wanting_nothing_is_certain() -> None:
    assert subset_survival_probability(4, 0, CONFIG) == 1.0


def test_impossible_when_desired_exceeds_pool() -> None:
    assert subset_survival_probability(2, 3, CONFIG) == 0.0
    assert subset_survival_probability(0, 1, CONFIG) == 0.0


def test_single_passive_probability_matches_expected_draw_size() -> None:
    """P(one specific passive survives) must equal E[N] / pool_size."""
    expected_draw = sum(
        w * (i + 1) for i, w in enumerate(CONFIG.inherited_count_weights)
    )
    for pool in range(len(CONFIG.inherited_count_weights), 9):
        assert subset_survival_probability(pool, 1, CONFIG) == pytest.approx(
            expected_draw / pool
        )


def test_all_four_from_a_four_pool_needs_the_four_roll() -> None:
    # Only a roll of 4 can carry all four across, so the answer is that weight.
    assert subset_survival_probability(4, 4, CONFIG) == pytest.approx(0.1)


def test_small_pools_are_capped_not_discarded() -> None:
    # With only two passives between the parents, any roll of 2 or more hands
    # over both; the probability is the weight of those rolls.
    assert subset_survival_probability(2, 2, CONFIG) == pytest.approx(0.6)


def test_probability_falls_as_the_pool_grows() -> None:
    previous = 1.0
    for pool in range(4, 12):
        current = subset_survival_probability(pool, 4, CONFIG)
        assert current < previous
        previous = current


def test_probability_falls_as_more_passives_are_wanted() -> None:
    values = [subset_survival_probability(8, k, CONFIG) for k in range(1, 5)]
    assert values == sorted(values, reverse=True)


def test_matches_a_monte_carlo_simulation() -> None:
    """Cross-check the closed form against a direct simulation of the rules."""
    rng = random.Random(20260913)
    pool = ["a", "b", "c", "d", "e", "f"]
    desired = {"a", "b"}
    weights = CONFIG.inherited_count_weights
    counts = list(range(1, len(weights) + 1))

    trials = 200_000
    hits = 0
    for _ in range(trials):
        draw = min(rng.choices(counts, weights=weights)[0], len(pool))
        if desired <= set(rng.sample(pool, draw)):
            hits += 1

    simulated = hits / trials
    exact = subset_survival_probability(len(pool), len(desired), CONFIG)
    assert simulated == pytest.approx(exact, abs=0.005)


def test_matches_exhaustive_enumeration() -> None:
    """Independent exact check: enumerate every draw instead of sampling.

    Given a roll of `size`, each subset of that size is equally likely, so the
    probability of a specific pair surviving is the fraction of those subsets
    that contain it, weighted by the roll. Computing it this way shares no code
    with the closed form, so agreement is real evidence rather than a tautology.
    """
    pool = ("a", "b", "c", "d", "e")
    desired = {"a", "b"}

    expected = 0.0
    for i, weight in enumerate(CONFIG.inherited_count_weights):
        size = min(i + 1, len(pool))
        subsets = list(combinations(pool, size))
        containing = sum(1 for subset in subsets if desired <= set(subset))
        expected += weight * containing / len(subsets)

    assert subset_survival_probability(len(pool), len(desired), CONFIG) == pytest.approx(
        expected
    )


def test_odds_report_attempts_needed() -> None:
    odds = odds_for_pair(["a"], ["a", "b"], ["c", "d"], CONFIG)
    assert odds.pool_size == 4
    assert odds.probability == pytest.approx(0.5)
    assert odds.expected_attempts == pytest.approx(2.0)
    # 90% confidence at p=0.5 needs 4 eggs: 1 - 0.5^4 = 0.9375.
    assert odds.attempts_for_90pct == 4
    assert odds.attempts_for_99pct == 7


def test_unavailable_passives_make_it_impossible() -> None:
    odds = odds_for_pair(["missing"], ["a"], ["b"], CONFIG)
    assert odds.probability == 0.0
    assert odds.unavailable == ("missing",)
    assert odds.expected_attempts is None
    assert not odds.is_possible


def test_duplicate_passives_across_parents_count_once() -> None:
    shared = odds_for_pair(["a"], ["a", "b"], ["a", "b"], CONFIG)
    assert shared.pool_size == 2


def test_route_annotation_tracks_carried_passives(planner) -> None:
    plan = planner.plan("apex", owned=["p10", "p1000"])
    engine = InheritanceEngine(CONFIG)
    annotated = engine.annotate_route(
        plan,
        desired=["fierce", "serene"],
        starting_passives={"p10": ["fierce"], "p1000": ["serene"]},
    )

    assert len(annotated.steps) == len(plan.steps)
    assert annotated.achieved == ("fierce", "serene")
    assert annotated.unreachable_passives == ()
    assert 0.0 < annotated.clean_run_probability <= 1.0
    assert annotated.total_expected_attempts >= len(plan.steps)


def test_route_annotation_flags_passives_no_parent_has(planner) -> None:
    plan = planner.plan("apex", owned=["p10", "p1000"])
    engine = InheritanceEngine(CONFIG)
    annotated = engine.annotate_route(
        plan, desired=["fierce"], starting_passives={}
    )
    # Nobody in the starting set carries it, so the route cannot deliver it.
    assert annotated.achieved == ()
    assert annotated.unreachable_passives == ("fierce",)


def test_config_rejects_weights_that_do_not_sum_to_one() -> None:
    with pytest.raises(ValueError, match="must sum to 1"):
        InheritanceConfig(inherited_count_weights=(0.5, 0.2))


def test_config_rejects_negative_weights() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        InheritanceConfig(inherited_count_weights=(1.5, -0.5))


def test_display_names_resolve_to_ids(planner, index) -> None:
    """Names and ids must be interchangeable, as they are everywhere else.

    A pal's internal id rarely matches its display name (Lamball is
    `sheepball`), so an unresolved name would silently look like a parent that
    carries nothing -- a zero-probability route with no visible cause.
    """
    plan = planner.plan("APEX", owned=["p10", "p1000"])
    engine = InheritanceEngine(CONFIG, index)
    annotated = engine.annotate_route(
        plan,
        desired=["Fierce"],
        starting_passives={"P10": ["Fierce"]},
    )
    assert annotated.desired == ("fierce",)
    assert annotated.achieved == ("fierce",)


def test_without_an_index_ids_are_still_required(planner) -> None:
    plan = planner.plan("apex", owned=["p10", "p1000"])
    engine = InheritanceEngine(CONFIG)
    annotated = engine.annotate_route(
        plan, desired=["Fierce"], starting_passives={"P10": ["Fierce"]}
    )
    # No resolver, so the display names never match the route's ids. The result
    # is honest about it rather than pretending the passive was delivered.
    assert annotated.achieved == ()
    assert annotated.unreachable_passives == ("Fierce",)
