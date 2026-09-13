"""The breeding engine.

Resolution order for a pair, highest priority first:

1. A special combo entry, which overrides everything.
2. Same species on both sides, which always breeds true.
3. The rank formula: ``target = floor((rank_a + rank_b + 1) / 2)``, then the
   breedable child whose ``combi_rank`` is nearest to ``target``.

The formula and the special-combo list are game rules; the tie-break when two
children sit equidistant from ``target`` is the one place the community data
is thin, so it is centralised in `PalIndex.nearest_child` and documented there
rather than being scattered through this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .dataset import PalIndex
from .models import Pal


class BreedingError(ValueError):
    """Raised when a pair cannot be bred at all."""


@dataclass(frozen=True, slots=True)
class BreedingResult:
    """The outcome of one pair, with the reason attached.

    `rule` matters to callers: a special-combo result is exact game data, while
    a formula result is only as good as the two ranks that fed it.
    """

    parent_a: str
    parent_b: str
    child: str
    rule: str  # "special" | "same_species" | "formula"
    target_rank: int | None = None


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


class BreedingEngine:
    """Resolves pairs to children, and inverts that relation.

    The full N×N table is materialised lazily on first use. At a few hundred
    pals that is tens of thousands of entries, which is cheap to hold and turns
    every later query, including the route search, into a dict lookup.
    """

    def __init__(self, index: PalIndex) -> None:
        self.index = index

    def breed(self, parent_a: str, parent_b: str) -> BreedingResult:
        """Resolve one pair. Raises `BreedingError` if the pair is invalid."""
        pal_a = self.index.get(parent_a)
        pal_b = self.index.get(parent_b)
        if pal_a is None:
            raise BreedingError(f"unknown pal: {parent_a!r}")
        if pal_b is None:
            raise BreedingError(f"unknown pal: {parent_b!r}")
        return self._breed_pals(pal_a, pal_b)

    def _breed_pals(self, pal_a: Pal, pal_b: Pal) -> BreedingResult:
        key = _pair_key(pal_a.id, pal_b.id)

        special = self.index.special_combos.get(key)
        if special is not None:
            return BreedingResult(key[0], key[1], special, "special")

        if not pal_a.is_breedable_parent:
            raise BreedingError(f"{pal_a.id!r} cannot be used as a parent")
        if not pal_b.is_breedable_parent:
            raise BreedingError(f"{pal_b.id!r} cannot be used as a parent")

        # A pair of the same species always breeds true, regardless of what the
        # rank formula would otherwise land on. For a pal that can parent but
        # never hatch, that leaves no valid outcome at all -- the formula does
        # not get a second go, because the same-species rule has already won.
        if pal_a.id == pal_b.id:
            if not pal_a.breedable_as_child:
                raise BreedingError(
                    f"{pal_a.id!r} breeds true but cannot be hatched, so a pair "
                    f"of them has no valid offspring"
                )
            return BreedingResult(key[0], key[1], pal_a.id, "same_species")

        target = (pal_a.combi_rank + pal_b.combi_rank + 1) // 2  # type: ignore[operator]
        child = self.index.nearest_child(target)
        if child is None:
            raise BreedingError("dataset contains no breedable children")
        return BreedingResult(key[0], key[1], child.id, "formula", target_rank=target)

    def try_breed(self, parent_a: str, parent_b: str) -> BreedingResult | None:
        """Like `breed`, but returns None instead of raising on an invalid pair."""
        try:
            return self.breed(parent_a, parent_b)
        except BreedingError:
            return None

    @cached_property
    def table(self) -> dict[tuple[str, str], str]:
        """Every valid unordered pair mapped to its child."""
        result: dict[tuple[str, str], str] = {}
        parents = self.index.parent_pool
        for i, pal_a in enumerate(parents):
            for pal_b in parents[i:]:
                try:
                    outcome = self._breed_pals(pal_a, pal_b)
                except BreedingError:
                    continue
                result[_pair_key(pal_a.id, pal_b.id)] = outcome.child

        # Special combos may name parents outside the ordinary parent pool
        # (a unique that can breed but never be hatched, say), so fold them in
        # separately rather than relying on the loop above to have seen them.
        for key, child in self.index.special_combos.items():
            result[key] = child
        return result

    @cached_property
    def parents_of(self) -> dict[str, tuple[tuple[str, str], ...]]:
        """Inverse of `table`: child id -> every pair that produces it."""
        inverse: dict[str, list[tuple[str, str]]] = {}
        for pair, child in self.table.items():
            inverse.setdefault(child, []).append(pair)
        return {child: tuple(sorted(pairs)) for child, pairs in inverse.items()}

    def pairs_producing(self, child_id: str) -> tuple[tuple[str, str], ...]:
        """Every parent pair that yields `child_id`."""
        if child_id not in self.index.pals:
            raise BreedingError(f"unknown pal: {child_id!r}")
        return self.parents_of.get(child_id, ())

    def children_of(self, pal_id: str) -> dict[str, str]:
        """Map of partner id -> resulting child, for every partner of `pal_id`."""
        if pal_id not in self.index.pals:
            raise BreedingError(f"unknown pal: {pal_id!r}")
        out: dict[str, str] = {}
        for partner in self.index.parent_pool:
            child = self.table.get(_pair_key(pal_id, partner.id))
            if child is not None:
                out[partner.id] = child
        return out

    @cached_property
    def unreachable(self) -> tuple[str, ...]:
        """Pals no pair can produce.

        Useful as a data-quality signal: a pal that should be breedable showing
        up here usually means a missing rank or a missing special combo.
        """
        producible = set(self.parents_of)
        return tuple(
            sorted(p.id for p in self.index.dataset.pals if p.id not in producible)
        )
