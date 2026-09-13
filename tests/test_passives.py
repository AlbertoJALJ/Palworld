"""Passive scoring and loadout selection."""

from __future__ import annotations

import pytest

from palworld_api.passives import (
    BASE_PROFILE,
    COMBAT_PROFILE,
    GoalProfile,
    PassiveEngine,
)


def _ids(recommendation) -> set[str]:
    return {p.passive_id for p in recommendation.passives}


def test_combat_prefers_attack_passives(passive_engine: PassiveEngine) -> None:
    ids = _ids(passive_engine.recommend("p10", "combat"))
    assert "fierce" in ids


def test_base_and_combat_disagree(passive_engine: PassiveEngine) -> None:
    combat = _ids(passive_engine.recommend("p10", "combat"))
    base = _ids(passive_engine.recommend("p10", "base"))
    assert "fierce" in combat and "fierce" not in base
    assert "artisan" in base and "artisan" not in combat


def test_only_one_member_of_an_exclusive_group(passive_engine: PassiveEngine) -> None:
    # fierce (+20 attack) and brave (+10 attack) share a group; the weaker one
    # must never take a slot the stronger one could use.
    ids = _ids(passive_engine.recommend("p10", "combat"))
    assert "fierce" in ids
    assert "brave" not in ids


def test_detrimental_passives_are_never_selected(passive_engine: PassiveEngine) -> None:
    for goal in ("combat", "base"):
        assert "clumsy" not in _ids(passive_engine.recommend("p10", goal))


def test_conditional_passives_are_discounted(passive_engine: PassiveEngine) -> None:
    recommendation = passive_engine.recommend("p10", "combat")
    nocturnal = next(p for p in recommendation.passives if p.passive_id == "nocturnal")
    # +30 attack, but only at night, so it scores below a flat +20.
    assert nocturnal.unconditional_score == 30.0
    assert nocturnal.score == 15.0
    fierce = next(p for p in recommendation.passives if p.passive_id == "fierce")
    assert fierce.score > nocturnal.score


def test_condition_reliability_is_tunable(passive_engine: PassiveEngine) -> None:
    always_night = GoalProfile(
        name="night", weights=COMBAT_PROFILE.weights, condition_reliability=1.0
    )
    recommendation = passive_engine.recommend("p10", always_night)
    nocturnal = next(p for p in recommendation.passives if p.passive_id == "nocturnal")
    assert nocturnal.score == 30.0


def test_restricted_passives_are_offered_only_to_their_pal(
    passive_engine: PassiveEngine,
) -> None:
    assert "apex_only" in _ids(passive_engine.recommend("apex", "combat"))
    generic = passive_engine.recommend("p10", "combat")
    assert "apex_only" not in _ids(generic)
    assert "apex_only" in generic.excluded_restricted


def test_work_passives_are_worthless_on_a_pal_that_cannot_work(
    passive_engine: PassiveEngine,
) -> None:
    # p10 mines at level 4; p50 has no work suitabilities at all.
    worker = passive_engine.recommend("p10", "base")
    idler = passive_engine.recommend("p50", "base")
    assert "artisan" in _ids(worker)
    assert "artisan" not in _ids(idler)


def test_sanity_reduction_counts_as_a_gain(passive_engine: PassiveEngine) -> None:
    # serene is -15 sanity loss, which must score positively, and more so for
    # a base pal than a fighter.
    base = passive_engine.recommend("p10", "base")
    combat = passive_engine.recommend("p10", "combat")
    base_serene = next(p for p in base.passives if p.passive_id == "serene")
    combat_serene = next(p for p in combat.passives if p.passive_id == "serene")
    assert base_serene.score > 0
    assert base_serene.score > combat_serene.score


def test_never_exceeds_the_slot_limit(passive_engine: PassiveEngine) -> None:
    for goal in ("combat", "base"):
        recommendation = passive_engine.recommend("apex", goal)
        assert len(recommendation.passives) <= 4


def test_slot_limit_is_respected_when_binding(passive_engine: PassiveEngine) -> None:
    two_slots = GoalProfile(name="two", weights=COMBAT_PROFILE.weights, max_slots=2)
    recommendation = passive_engine.recommend("apex", two_slots)
    assert len(recommendation.passives) == 2
    # It must keep the two best, not simply the first two it saw.
    assert {p.passive_id for p in recommendation.passives} == {"apex_only", "fierce"}


def test_results_are_sorted_by_score(passive_engine: PassiveEngine) -> None:
    scores = [p.score for p in passive_engine.recommend("apex", "combat").passives]
    assert scores == sorted(scores, reverse=True)


def test_selection_is_optimal_against_brute_force(passive_engine: PassiveEngine) -> None:
    """The closed-form selection must match an exhaustive search."""
    from itertools import combinations

    pal = passive_engine.index.require("apex")
    for profile in (BASE_PROFILE, COMBAT_PROFILE):
        scored, _ = passive_engine.candidates(pal, profile)
        chosen = passive_engine.select(scored, profile.max_slots)
        actual = sum(p.score for p in chosen)

        best = 0.0
        for size in range(profile.max_slots + 1):
            for combo in combinations(scored, size):
                groups = [p.exclusive_group for p in combo if p.exclusive_group]
                if len(groups) != len(set(groups)):
                    continue
                best = max(best, sum(p.score for p in combo))
        assert actual == pytest.approx(best), profile.name


def test_unknown_pal_and_goal_raise(passive_engine: PassiveEngine) -> None:
    with pytest.raises(KeyError, match="unknown pal"):
        passive_engine.recommend("nope", "combat")
    with pytest.raises(KeyError, match="unknown goal"):
        passive_engine.recommend("p10", "sandwich")
