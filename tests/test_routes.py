"""Route planning over the breeding hypergraph."""

from __future__ import annotations

import pytest

from palworld_api.routes import RouteError, RoutePlanner, Strategy


def _assert_well_ordered(plan, planner: RoutePlanner) -> None:
    """Every step must be performable when it is reached, in listed order."""
    have = set(plan.owned)
    for step in plan.steps:
        assert step.parent_a in have, f"{step.parent_a} not available yet"
        assert step.parent_b in have, f"{step.parent_b} not available yet"
        pair = (step.parent_a, step.parent_b)
        if pair[0] > pair[1]:
            pair = (pair[1], pair[0])
        assert step.child in planner.engine.outcomes[pair]
        have.add(step.child)
    if plan.steps:
        assert plan.target in have


def test_owning_the_target_alone_is_not_enough_to_breed_it(planner: RoutePlanner) -> None:
    # Whether you already have the target is irrelevant to this tool: it
    # always tries to find an actual recipe, so giving it nothing else to
    # breed from is a real error, not a free pass.
    with pytest.raises(RouteError, match="is the only pal given"):
        planner.plan("apex", owned=["apex"])


def test_owning_the_target_does_not_short_circuit_a_real_recipe(
    planner: RoutePlanner,
) -> None:
    # p100 is wild-obtainable (so a plausible member of "owned"), and also
    # breedable from p50 + p200. Listing it as owned alongside real breeding
    # stock must not suppress the recipe -- ownership is not this tool's
    # business, a recipe is.
    plan = planner.plan("p100", owned=["p100", "p50", "p200"])
    assert plan.steps
    assert plan.steps[-1].child == "p100"
    _assert_well_ordered(plan, planner)


def test_single_step_route(planner: RoutePlanner) -> None:
    plan = planner.plan("apex", owned=["p10", "p50"])
    assert plan.generations == 1
    assert plan.distinct_steps == 1
    assert plan.steps[0].child == "apex"
    _assert_well_ordered(plan, planner)


def test_multi_step_route_prefers_the_special_combo(planner: RoutePlanner) -> None:
    # Climbing the ranks one at a time would take five generations; going up to
    # p800 and using the p800+p1000 special combo takes three.
    plan = planner.plan("apex", owned=["p10", "p1000"])
    assert plan.generations == 3
    assert plan.distinct_steps == 3
    assert plan.steps[-1].child == "apex"
    assert plan.steps[-1].rule == "special"
    _assert_well_ordered(plan, planner)


def test_route_is_reproducible(planner: RoutePlanner) -> None:
    first = planner.plan("apex", owned=["p10", "p1000"])
    second = planner.plan("apex", owned=["p10", "p1000"])
    assert first.steps == second.steps


def test_intermediates_exclude_the_target(planner: RoutePlanner) -> None:
    plan = planner.plan("apex", owned=["p10", "p1000"])
    assert "apex" not in plan.intermediates
    assert set(plan.intermediates) == {s.child for s in plan.steps[:-1]}


def test_tree_strategy_never_beats_generation_count(planner: RoutePlanner) -> None:
    owned = ["p10", "p1000"]
    generations = planner.plan("apex", owned=owned, strategy=Strategy.GENERATIONS)
    tree = planner.plan("apex", owned=owned, strategy=Strategy.TREE)
    # A tree plan counts shared work twice, so it can never need fewer rounds.
    assert tree.generations >= generations.generations
    _assert_well_ordered(tree, planner)


def test_default_owned_excludes_non_wild_pals(planner: RoutePlanner) -> None:
    assert "apex" not in planner.default_owned()
    plan = planner.plan("apex")
    assert plan.steps
    _assert_well_ordered(plan, planner)


def test_default_owned_still_finds_a_recipe_for_a_wild_target(
    planner: RoutePlanner,
) -> None:
    # p100 IS in the default owned set (wild-obtainable); planning a route to
    # it must not come back empty just because you could also go catch one.
    assert "p100" in planner.default_owned()
    plan = planner.plan("p100")
    assert plan.steps
    assert plan.steps[-1].child == "p100"
    _assert_well_ordered(plan, planner)


def test_unknown_target_raises(planner: RoutePlanner) -> None:
    with pytest.raises(RouteError, match="unknown pal"):
        planner.plan("nope")


def test_unknown_owned_pal_raises(planner: RoutePlanner) -> None:
    with pytest.raises(RouteError, match="unknown pal in owned set"):
        planner.plan("apex", owned=["nope"])


def test_empty_owned_set_raises(planner: RoutePlanner) -> None:
    with pytest.raises(RouteError, match="owned set is empty"):
        planner.plan("apex", owned=[])


def test_unreachable_target_raises(planner: RoutePlanner) -> None:
    with pytest.raises(RouteError, match="not reachable"):
        planner.plan("rankless", owned=["p10", "p50"])


def test_generation_cap_can_make_a_target_unreachable(planner: RoutePlanner) -> None:
    with pytest.raises(RouteError, match="not reachable"):
        planner.plan("apex", owned=["p10", "p1000"], max_generations=1)


def test_resolves_targets_by_display_name(planner: RoutePlanner) -> None:
    by_id = planner.plan("apex", owned=["p10", "p50"])
    by_name = planner.plan("APEX", owned=["p10", "p50"])
    assert by_id.steps == by_name.steps


def test_reachable_from_reports_generation_counts(planner: RoutePlanner) -> None:
    reachable = planner.reachable_from(["p10", "p1000"])
    assert reachable["p10"] == 0
    assert reachable["p1000"] == 0
    assert reachable["p400"] == 1
    assert reachable["apex"] == 3


def test_reachable_costs_match_planned_routes(planner: RoutePlanner) -> None:
    owned = ["p10", "p1000"]
    for pal_id, generations in planner.reachable_from(owned).items():
        if generations == 0:
            continue
        assert planner.plan(pal_id, owned=owned).generations == generations


def test_alternatives_are_all_actually_reachable(planner: RoutePlanner) -> None:
    reachable = planner.reachable_from(["p10", "p1000"])
    for pair in planner.alternatives("apex", owned=["p10", "p1000"]):
        assert pair[0] in reachable and pair[1] in reachable
        assert "apex" in planner.engine.outcomes[pair]
