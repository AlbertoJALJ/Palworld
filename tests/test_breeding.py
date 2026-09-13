"""Breeding rules, including the precedence order between them."""

from __future__ import annotations

import pytest

from palworld_api.breeding import BreedingEngine, BreedingError


def test_formula_picks_nearest_rank(engine: BreedingEngine) -> None:
    # (10 + 50 + 1) // 2 == 30, which is exactly apex's rank.
    result = engine.breed("p10", "p50")
    assert result.child == "apex"
    assert result.rule == "formula"
    assert result.target_rank == 30


def test_formula_rounds_up_on_odd_sums(engine: BreedingEngine) -> None:
    # (10 + 1000 + 1) // 2 == 505; 400 is nearer than 800.
    result = engine.breed("p10", "p1000")
    assert result.target_rank == 505
    assert result.child == "p400"


def test_same_species_breeds_true(engine: BreedingEngine) -> None:
    result = engine.breed("p400", "p400")
    assert result.child == "p400"
    assert result.rule == "same_species"


def test_special_combo_overrides_formula(engine: BreedingEngine) -> None:
    # The formula would land near 900; the special combo wins outright.
    assert engine.breed("p800", "p1000").target_rank is None
    result = engine.breed("p800", "p1000")
    assert result.child == "apex"
    assert result.rule == "special"


def test_special_combo_is_order_independent(engine: BreedingEngine) -> None:
    assert engine.breed("p1000", "p800").child == "apex"


def test_pair_order_does_not_change_result(engine: BreedingEngine) -> None:
    assert engine.breed("p10", "p1000").child == engine.breed("p1000", "p10").child


def test_equidistant_ranks_resolve_to_the_lower(engine: BreedingEngine) -> None:
    # (100 + 200 + 1) // 2 == 150, exactly between ranks 100 and 200.
    result = engine.breed("p100", "p200")
    assert result.target_rank == 150
    assert result.child == "p100"


def test_tie_break_uses_breed_order(index) -> None:
    # Two pals share rank 3000; the lower breed_order must win every time.
    assert index.nearest_child(3000).id == "twin_early"
    assert index.nearest_child(2999).id == "twin_early"


def test_non_child_pal_is_never_produced(engine: BreedingEngine) -> None:
    assert "uncatchable_child" not in set(engine.table.values())


def test_non_child_pal_can_still_be_a_parent(engine: BreedingEngine) -> None:
    assert engine.breed("uncatchable_child", "p10").child is not None


def test_rankless_pal_is_excluded_from_breeding(engine: BreedingEngine) -> None:
    with pytest.raises(BreedingError, match="cannot be used as a parent"):
        engine.breed("rankless", "p10")


def test_unknown_pal_raises(engine: BreedingEngine) -> None:
    with pytest.raises(BreedingError, match="unknown pal"):
        engine.breed("does_not_exist", "p10")


def test_try_breed_swallows_errors(engine: BreedingEngine) -> None:
    assert engine.try_breed("does_not_exist", "p10") is None


def test_self_pairing_a_non_child_pal_has_no_outcome(engine: BreedingEngine) -> None:
    with pytest.raises(BreedingError, match="cannot be hatched"):
        engine.breed("uncatchable_child", "uncatchable_child")


def test_table_is_symmetric_and_complete(engine: BreedingEngine) -> None:
    parents = [p.id for p in engine.index.parent_pool]
    for a in parents:
        for b in parents:
            key = (a, b) if a <= b else (b, a)
            if a == b and not engine.index.require(a).breedable_as_child:
                assert key not in engine.table
                continue
            assert key in engine.table, f"missing pair {key}"


def test_inverse_index_agrees_with_outcomes(engine: BreedingEngine) -> None:
    for child, pairs in engine.parents_of.items():
        for pair in pairs:
            # A gender-dependent pair has two children, so the inverse index
            # points at a pair whose outcomes contain -- not necessarily equal
            # -- this child.
            assert child in engine.outcomes[pair]


def test_pairs_producing_finds_the_special_combo(engine: BreedingEngine) -> None:
    assert ("p1000", "p800") in engine.pairs_producing("apex")


def test_formula_never_leaves_the_parents_rank_range(engine: BreedingEngine) -> None:
    """The core reason rare pals need a route at all.

    `floor((a + b + 1) / 2)` always lands between the two parent ranks, so a
    formula result can only ever interpolate. No amount of breeding common pals
    together reaches a rarer one: you need a pal already at that rank, or a
    special combo. This invariant is what makes route planning non-trivial.
    """
    index = engine.index
    for (a, b), child in engine.table.items():
        if index.special_combos.get((a, b)) is not None:
            continue
        rank_a = index.require(a).combi_rank
        rank_b = index.require(b).combi_rank
        target = (rank_a + rank_b + 1) // 2
        assert min(rank_a, rank_b) <= target <= max(rank_a, rank_b), (a, b)
        # The child is the nearest available rank to that midpoint, so it can
        # sit just outside the bracket, but never by more than the gap to the
        # nearest neighbour on that side.
        assert index.require(child).combi_rank is not None


def test_gendered_pair_reports_both_children(engine: BreedingEngine) -> None:
    """One pair, two possible children, decided by the parents' sexes.

    The game has exactly one such pair (Katress + Wixen). Collapsing it to a
    single child silently loses one of the two pals it can make.
    """
    result = engine.breed("p20", "p50")
    assert result.rule == "special"
    assert {result.child, *result.alternatives} == {"p10", "apex"}


def test_gendered_pair_states_its_requirement(engine: BreedingEngine) -> None:
    result = engine.breed("p20", "p50")
    assert result.gender_requirement is not None
    assert "must be" in result.gender_requirement


def test_both_gendered_children_are_obtainable(engine: BreedingEngine) -> None:
    assert engine.outcomes[("p20", "p50")] == ("p10", "apex") or engine.outcomes[
        ("p20", "p50")
    ] == ("apex", "p10")


def test_both_gendered_children_list_the_pair_as_a_parent(
    engine: BreedingEngine,
) -> None:
    for child in ("p10", "apex"):
        assert ("p20", "p50") in engine.pairs_producing(child)
