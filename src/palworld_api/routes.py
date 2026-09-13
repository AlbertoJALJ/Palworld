"""Route planning: how to get from the pals you have to the pal you want.

Breeding is not an ordinary graph problem. An edge needs *two* sources at once,
which makes this a shortest-hyperpath search over hyperedges ``(a, b) -> c``.
Ordinary Dijkstra does not apply, but Knuth's generalisation does, provided the
cost function is monotone and superior (``f(x, y) >= max(x, y)``). Both cost
functions here satisfy that, so the search is still correct and near-linear.

The other thing that makes this tractable: breeding in Palworld does not consume
the parents, so the set of species you own only ever grows. That monotonicity is
what lets a single pass finalise each pal once.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from heapq import heappop, heappush

from .breeding import BreedingEngine, _pair_key
from .dataset import PalIndex


class Strategy(StrEnum):
    """How to measure the cost of a route.

    GENERATIONS minimises the number of breeding *rounds*, treating work on the
    two parent lines as parallel. It is what you want when planning elapsed time,
    since you can raise both lines at once.

    TREE minimises the total number of breeding operations counted as a tree,
    i.e. without reusing a shared intermediate. It is pessimistic but it is what
    you want when eggs, not time, are the scarce resource.
    """

    GENERATIONS = "generations"
    TREE = "tree"


class RouteError(ValueError):
    """Raised when a route cannot be planned."""


@dataclass(frozen=True, slots=True)
class BreedStep:
    parent_a: str
    parent_b: str
    child: str
    rule: str
    generation: int


@dataclass(frozen=True, slots=True)
class RoutePlan:
    """A concrete plan, ordered so every step's parents already exist."""

    target: str
    steps: tuple[BreedStep, ...]
    strategy: Strategy
    owned: tuple[str, ...]
    generations: int
    distinct_steps: int
    intermediates: tuple[str, ...] = ()
    already_owned: bool = False

    @property
    def is_empty(self) -> bool:
        return not self.steps


@dataclass(order=True, slots=True)
class _QueueItem:
    cost: int
    pal: str = field(compare=False)


class RoutePlanner:
    """Plans breeding routes over a `BreedingEngine`."""

    def __init__(self, index: PalIndex, engine: BreedingEngine | None = None) -> None:
        self.index = index
        self.engine = engine or BreedingEngine(index)

    def default_owned(self) -> tuple[str, ...]:
        """Every pal you could reasonably start from without breeding."""
        return tuple(
            sorted(p.id for p in self.index.dataset.pals if p.wild_obtainable)
        )

    def _resolve_owned(self, owned: Iterable[str] | None) -> set[str]:
        if owned is None:
            return set(self.default_owned())
        resolved: set[str] = set()
        for ref in owned:
            pal = self.index.resolve(ref)
            if pal is None:
                raise RouteError(f"unknown pal in owned set: {ref!r}")
            resolved.add(pal.id)
        if not resolved:
            raise RouteError("owned set is empty; nothing to breed from")
        return resolved

    def plan(
        self,
        target: str,
        owned: Iterable[str] | None = None,
        strategy: Strategy = Strategy.GENERATIONS,
        max_generations: int | None = None,
    ) -> RoutePlan:
        """Find the cheapest route to `target` from `owned`.

        Raises `RouteError` if the target is unknown or unreachable.
        """
        target_pal = self.index.resolve(target)
        if target_pal is None:
            raise RouteError(f"unknown pal: {target!r}")
        target_id = target_pal.id

        owned_ids = self._resolve_owned(owned)

        if target_id in owned_ids:
            return RoutePlan(
                target=target_id,
                steps=(),
                strategy=strategy,
                owned=tuple(sorted(owned_ids)),
                generations=0,
                distinct_steps=0,
                already_owned=True,
            )

        best, via, _ = self._search(
            owned_ids, strategy, max_generations, target_id=target_id
        )
        if target_id not in best:
            raise RouteError(
                f"{target_id!r} is not reachable by breeding from the given pals"
            )

        steps = self._reconstruct(target_id, via, owned_ids)
        generations = max((s.generation for s in steps), default=0)
        intermediates = tuple(
            s.child for s in steps if s.child != target_id
        )
        return RoutePlan(
            target=target_id,
            steps=steps,
            strategy=strategy,
            owned=tuple(sorted(owned_ids)),
            generations=generations,
            distinct_steps=len(steps),
            intermediates=intermediates,
        )

    def _search(
        self,
        owned_ids: set[str],
        strategy: Strategy,
        max_generations: int | None,
        target_id: str | None = None,
    ) -> tuple[dict[str, int], dict[str, tuple[str, str]], dict[str, int]]:
        """Knuth's generalised Dijkstra over the breeding hypergraph.

        Returns the tentative cost of each seen pal, the pair each was reached
        by, and the settled costs of the pals that were finalised. Only the
        settled costs are guaranteed optimal: when `target_id` is given the
        search stops early, leaving the rest of the frontier tentative.
        """
        combine = (
            (lambda x, y: max(x, y) + 1)
            if strategy is Strategy.GENERATIONS
            else (lambda x, y: x + y + 1)
        )

        outcomes = self.engine.outcomes
        best: dict[str, int] = {pal: 0 for pal in owned_ids}
        via: dict[str, tuple[str, str]] = {}
        finalized: dict[str, int] = {}
        heap: list[_QueueItem] = [_QueueItem(0, pal) for pal in sorted(owned_ids)]
        # Heapify by construction: all seeds share cost 0, so insertion order
        # already satisfies the heap property.

        while heap:
            item = heappop(heap)
            pal, cost = item.pal, item.cost
            if pal in finalized:
                continue
            finalized[pal] = cost

            if pal == target_id:
                break

            # Pair the newly finalised pal with every already-finalised one
            # (itself included). Every valid pair is therefore considered
            # exactly once, when its second member is finalised.
            for other, other_cost in finalized.items():
                # A pair usually has one child, but a gender-dependent pair has
                # two and both are obtainable, so relax towards every outcome.
                for child in outcomes.get(_pair_key(pal, other), ()):
                    if child in finalized:
                        continue
                    new_cost = combine(cost, other_cost)
                    if max_generations is not None and new_cost > max_generations:
                        continue
                    if new_cost < best.get(child, 1 << 30):
                        best[child] = new_cost
                        via[child] = (pal, other)
                        heappush(heap, _QueueItem(new_cost, child))

        return best, via, finalized

    def _reconstruct(
        self,
        target_id: str,
        via: dict[str, tuple[str, str]],
        owned_ids: set[str],
    ) -> tuple[BreedStep, ...]:
        """Walk the predecessor map into a deduplicated, ordered step list.

        A shared intermediate is emitted once, which is the whole point of
        reporting `distinct_steps` separately from the tree cost.
        """
        steps: list[BreedStep] = []
        depth: dict[str, int] = {pal: 0 for pal in owned_ids}
        emitted: set[str] = set()

        # Iterative post-order so a deep chain cannot blow the stack.
        stack: list[tuple[str, bool]] = [(target_id, False)]
        visiting: set[str] = set()
        while stack:
            node, expanded = stack.pop()
            if node in owned_ids or node in emitted:
                continue
            if expanded:
                pair = via[node]
                result = self.engine.breed(*pair)
                # `breed` reports the pair's primary child; this step may be the
                # gender-dependent alternative instead, which is still a
                # special-combo step but with a sex requirement attached.
                rule = result.rule
                if result.child != node:
                    rule = "special"
                generation = max(depth[pair[0]], depth[pair[1]]) + 1
                depth[node] = generation
                emitted.add(node)
                steps.append(
                    BreedStep(
                        parent_a=pair[0],
                        parent_b=pair[1],
                        child=node,
                        rule=rule,
                        generation=generation,
                    )
                )
                visiting.discard(node)
                continue

            if node not in via:
                raise RouteError(f"no known way to obtain {node!r}")
            if node in visiting:
                # The search never builds a cycle, but reconstructing one would
                # hang rather than fail, so refuse explicitly.
                raise RouteError(f"cyclic breeding route detected at {node!r}")
            visiting.add(node)
            stack.append((node, True))
            for parent in via[node]:
                if parent not in owned_ids and parent not in emitted:
                    stack.append((parent, False))

        steps.sort(key=lambda s: (s.generation, s.child))
        return tuple(steps)

    def reachable_from(
        self, owned: Iterable[str] | None = None, max_generations: int | None = None
    ) -> dict[str, int]:
        """Every pal reachable from `owned`, mapped to its generation count."""
        owned_ids = self._resolve_owned(owned)
        _, _, settled = self._search(
            owned_ids, Strategy.GENERATIONS, max_generations
        )
        return dict(sorted(settled.items(), key=lambda kv: (kv[1], kv[0])))

    def alternatives(
        self, target: str, owned: Iterable[str] | None = None, limit: int = 5
    ) -> tuple[tuple[str, str], ...]:
        """Final-step parent pairs for `target` whose parents are all reachable.

        Useful when the cheapest route needs a pal you would rather not chase;
        these are the other ways to close out the last step.
        """
        target_pal = self.index.resolve(target)
        if target_pal is None:
            raise RouteError(f"unknown pal: {target!r}")
        reachable = self.reachable_from(owned)
        viable = [
            pair
            for pair in self.engine.pairs_producing(target_pal.id)
            if pair[0] in reachable and pair[1] in reachable
        ]
        viable.sort(key=lambda p: (reachable[p[0]] + reachable[p[1]], p))
        return tuple(viable[:limit])
