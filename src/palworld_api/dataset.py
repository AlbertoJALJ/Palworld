"""Loading and indexing of a `Dataset`.

Kept separate from `models` so the models stay pure data and this module owns
the derived lookups (by id, by name, by rank) that the engines need to be fast.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from functools import cached_property
from pathlib import Path

from .models import Dataset, Pal, Passive, SpecialCombo


class DatasetError(RuntimeError):
    """Raised when a dataset is missing or fails validation."""


class PalIndex:
    """Read-only indexed view over a `Dataset`.

    Built once at startup. Every lookup the engines perform in a hot loop goes
    through a dict or a sorted array here rather than scanning the pal list.
    """

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.pals: dict[str, Pal] = {p.id: p for p in dataset.pals}
        self.passives: dict[str, Passive] = {p.id: p for p in dataset.passives}
        self._by_name = {p.name.casefold(): p for p in dataset.pals}
        self._by_passive_name = {p.name.casefold(): p for p in dataset.passives}

        # A pair can carry more than one combo when the outcome depends on
        # which parent is which sex, so index every combo, not just the last.
        self.combos_by_pair: dict[tuple[str, str], tuple[SpecialCombo, ...]] = {}
        for combo in dataset.special_combos:
            self.combos_by_pair.setdefault(combo.key, ())
            self.combos_by_pair[combo.key] += (combo,)

        # The gender-agnostic view the formula path needs: one child per pair,
        # preferring an ungendered combo when both kinds exist.
        self.special_combos: dict[tuple[str, str], str] = {
            pair: next(
                (c.child for c in combos if not c.is_gendered), combos[0].child
            )
            for pair, combos in self.combos_by_pair.items()
        }

        # Children the rank formula is allowed to land on, sorted by rank so the
        # nearest-rank lookup is a binary search instead of a linear scan.
        children = [p for p in dataset.pals if p.is_breedable_child]
        children.sort(key=lambda p: (p.combi_rank, self._tie_break(p)))
        self.child_pool: tuple[Pal, ...] = tuple(children)
        self._child_ranks: list[int] = [p.combi_rank for p in children]  # type: ignore[misc]

        self.parent_pool: tuple[Pal, ...] = tuple(
            p for p in dataset.pals if p.is_breedable_parent
        )

    @staticmethod
    def _tie_break(pal: Pal) -> tuple[int, str]:
        """Deterministic ordering for pals sitting equally close to a rank.

        Explicit `breed_order` wins; otherwise fall back to the numeric part of
        the paldeck number, then the id, so the result never depends on dict
        ordering or scrape order.
        """
        if pal.breed_order is not None:
            return (pal.breed_order, pal.id)
        if pal.paldeck:
            digits = "".join(c for c in pal.paldeck if c.isdigit())
            if digits:
                return (int(digits), pal.id)
        return (1 << 30, pal.id)

    def get(self, pal_id: str) -> Pal | None:
        return self.pals.get(pal_id)

    def require(self, pal_id: str) -> Pal:
        pal = self.pals.get(pal_id)
        if pal is None:
            raise KeyError(f"unknown pal id: {pal_id!r}")
        return pal

    def resolve(self, ref: str) -> Pal | None:
        """Look a pal up by id or by display name, case-insensitively."""
        return self.pals.get(ref) or self._by_name.get(ref.casefold())

    def resolve_passive(self, ref: str) -> Passive | None:
        return self.passives.get(ref) or self._by_passive_name.get(ref.casefold())

    def nearest_child(self, target_rank: int) -> Pal | None:
        """The breedable child whose rank is closest to `target_rank`.

        Ties are broken by `_tie_break`, which the sort order already encodes:
        when the distance is equal on both sides we take the lower rank, and
        within one rank the earliest tie-break key.
        """
        ranks = self._child_ranks
        if not ranks:
            return None

        pos = bisect_left(ranks, target_rank)
        if pos == 0:
            return self.child_pool[0]
        if pos == len(ranks):
            return self.child_pool[-1]

        # `pos` is the first index >= target; its left neighbour is the last
        # index < target. The winner is whichever is nearer, lower rank on a tie.
        lo, hi = self.child_pool[pos - 1], self.child_pool[pos]
        d_lo = target_rank - lo.combi_rank  # type: ignore[operator]
        d_hi = hi.combi_rank - target_rank  # type: ignore[operator]
        if d_hi < d_lo:
            return hi
        if d_lo < d_hi:
            # Several pals can share the lower rank; the sort put the winning
            # tie-break first, so walk back to the start of that rank block.
            return self._first_with_rank(pos - 1)
        return self._first_with_rank(pos - 1)

    def _first_with_rank(self, index: int) -> Pal:
        rank = self._child_ranks[index]
        while index > 0 and self._child_ranks[index - 1] == rank:
            index -= 1
        return self.child_pool[index]

    @cached_property
    def unverified_pals(self) -> tuple[str, ...]:
        """Pals whose numbers should not be trusted, surfaced by the API."""
        return tuple(
            p.id
            for p in self.dataset.pals
            if p.provenance is None or not p.provenance.verified
        )


def load_dataset(path: str | Path) -> Dataset:
    """Read and validate a dataset JSON document."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(
            f"dataset not found at {path}. Run `python -m palworld_api.ingest.cli "
            f"scrape` to build one, or point PALWORLD_DATASET at an existing file."
        )
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetError(f"dataset at {path} is not valid JSON: {exc}") from exc
    try:
        return Dataset.model_validate(raw)
    except Exception as exc:
        raise DatasetError(f"dataset at {path} failed validation: {exc}") from exc


def load_index(path: str | Path) -> PalIndex:
    return PalIndex(load_dataset(path))
